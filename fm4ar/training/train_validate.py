"""
Methods to train (or validate) a given model for one epoch.
"""

import time
from typing import TYPE_CHECKING, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fm4ar.torchutils.schedulers import perform_scheduler_step
from fm4ar.torchutils.general import check_for_nans
from fm4ar.training.stages import StageConfig
from fm4ar.utils.tracking import LossInfo

if TYPE_CHECKING:  # pragma: no cover
    from fm4ar.models.base import Base
    from fm4ar.models.fmpe import FMPEModel


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Move a batch of data to the given device.

    This function also separates `theta` from the `context` (to make
    sure no model every accidentally receives `theta` as an input).

    Args:
        batch: A dictionary containing the batch data.
        device: The device to which to move the data.

    Returns:
        A 2-tuple, `(theta, context)`, where `theta` is are the target
        parameters and `context` is the context dict.
    """

    # Move everthing to the device first
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }

    # Separate theta from context
    theta = batch.pop("theta")
    context = batch

    return theta, context


def train_epoch(
    model: "Base",
    dataloader: DataLoader,
    stage_config: StageConfig,
) -> float:
    """
    Train the posterior model for one epoch.

    Args:
        model: Model to train.
        dataloader: Dataloader for training data.
        stage_config: Configuration for the current training stage.

    Returns:
        Average loss over the epoch.
    """

    # Define shortcuts
    gradient_clipping_config = stage_config.gradient_clipping

    # Check if we can use automatic mixed precision
    if stage_config.use_amp and model.device == torch.device("cpu"):
        raise RuntimeError(  # pragma: no cover
            "Don't use automatic mixed precision on CPU!"
        )

    # Ensure that the neural net is in training mode
    model.network.train()

    # Set up a LossInfo object to keep track of the loss and times
    loss_info = LossInfo(
        epoch=model.epoch,
        len_dataset=len(dataloader.dataset),  # type: ignore
        batch_size=int(dataloader.batch_size),  # type: ignore
        mode="Train",
        print_freq=1,
    )

    # Create scaler for automatic mixed precision
    scaler = GradScaler(enabled=stage_config.use_amp)

    # Iterate over the batches
    batch: dict[str, torch.Tensor]
    for batch_idx, batch in enumerate(dataloader):

        loss_info.update_timer()

        # Move data to device
        theta, context = move_batch_to_device(batch, model.device)

        # Reset gradients
        model.optimizer.zero_grad(set_to_none=True)

        # No automatic mixed precision
        if not stage_config.use_amp:

            # Compute loss and backpropagate
            loss = model.loss(
                theta=theta,
                context=context,
                **stage_config.loss_kwargs,
            )
            check_for_nans(loss, "train loss")
            loss.backward()  # type: ignore

            # Clip gradients if desired
            if gradient_clipping_config.enabled:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.network.parameters(),
                    max_norm=gradient_clipping_config.max_norm,
                    norm_type=gradient_clipping_config.norm_type,
                )

            # Take a step with the optimizer
            model.optimizer.step()

        # With automatic mixed precision (default)
        # This cannot be tested at the moment because it requires a GPU
        else:  # pragma: no cover

            # Compute loss and backpropagate
            # Note: Backward passes under autocast are not recommended
            with autocast():
                loss = model.loss(
                    theta=theta,
                    context=context,
                    **stage_config.loss_kwargs,
                )
                check_for_nans(loss, "train loss")
            scaler.scale(loss).backward()  # type: ignore

            # Clip gradients if desired
            if gradient_clipping_config.enabled:
                scaler.unscale_(model.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.network.parameters(),
                    max_norm=gradient_clipping_config.max_norm,
                    norm_type=gradient_clipping_config.norm_type,
                )

            # Take a step with the optimizer
            scaler.step(model.optimizer)
            scaler.update()

        # Take a step with the learning rate scheduler after each batch.
        # This is required, e.g., for the OneCycleLR scheduler.
        perform_scheduler_step(
            scheduler=model.scheduler,
            loss=None,
            end_of="batch",
        )

        # Update loss for history and logging
        loss_info.update(loss.detach().item(), n=len(theta))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def validate_epoch(
    model: Union["Base", "FMPEModel"],  # required because of log_prob_batch()
    dataloader: DataLoader,
    stage_config: StageConfig,
) -> tuple[float, float | None]:
    """
    Perform one validation epoch for the given model.

    Args:
        model: Model to validate.
        dataloader: Dataloader for validation data.
        stage_config: Configuration for the current training stage.

    Returns:
        A 2-tuple, consisting of:
        (1) The average validation loss over the epoch.
        (2) The average log probability of the true parameter values.
    """

    # TODO: Maybe we also want to use AMP here to speed up the logprob stuff?

    # Ensure that the neural net is in evaluation mode
    model.network.eval()

    # -------------------------------------------------------------------------
    # Compute validation loss
    # -------------------------------------------------------------------------

    # We first compute only the validation loss
    # We don't need to compute gradients for the validation set
    with torch.no_grad() and autocast(enabled=stage_config.use_amp):

        # Set up a LossInfo object to keep track of the loss and times
        loss_info = LossInfo(
            epoch=model.epoch,
            len_dataset=len(dataloader.dataset),  # type: ignore
            batch_size=dataloader.batch_size,  # type: ignore
            mode="Validate",
            print_freq=1,
        )

        # Iterate over the batches
        batch: dict[str, torch.Tensor]
        for batch_idx, batch in enumerate(dataloader):

            loss_info.update_timer()

            # Move data to device
            theta, context = move_batch_to_device(batch, model.device)

            # Compute validation loss
            loss = model.loss(
                theta=theta,
                context=context,
                **stage_config.loss_kwargs,
            )
            check_for_nans(loss, "validation loss")

            # Update loss for history and logging
            loss_info.update(loss.item(), len(theta))
            loss_info.print_info(batch_idx)

    # Select the average validation loss
    avg_loss = loss_info.get_avg()

    # -------------------------------------------------------------------------
    # Optionally: Compute log prob
    # -------------------------------------------------------------------------

    # Optionally, we now also compute the average log probability of the true
    # parameter values. This is relatively expensive for flow matching models,
    # so we only do this for a subset of a single batch.
    if (
        stage_config.logprob_evaluation.interval is not None
        and (model.epoch - 1) % stage_config.logprob_evaluation.interval == 0
    ):

        evaluation_start = time.time()
        print("Evaluating log probability...", end=" ", flush=True)

        # Get the first batch of the dataloader and move it to the device
        batch = next(iter(dataloader))
        theta, context = move_batch_to_device(batch, model.device)

        # Check the `model` is an FMPE model and select any additional kwargs
        # for the ODE solver that we might need. Note: We cannot directly use
        # `isinstance` here, since this would create a circular import.
        is_fmpe_model = model.__class__.__name__ == "FMPEModel"
        extra_kwargs = (
            {} if not is_fmpe_model
            else stage_config.logprob_evaluation.ode_solver.dict()
        )

        # Compute logprob of the first `n_samples` samples of the batch
        # Note: Trying to speed up this part with AMP has not been successful
        # so far and has mostly resulted in out-of-memory errors...
        n_samples = stage_config.logprob_evaluation.n_samples
        with torch.no_grad():
            logprob = model.log_prob_batch(
                theta=theta[:n_samples],
                context={k: v[:n_samples] for k, v in context.items()},
                **extra_kwargs,
            )

        # Compute the average log probability of the samples
        avg_logprob = float(logprob.mean().item())

        # Print the time it took to compute the log probability
        evaluation_time = time.time() - evaluation_start
        print(f"Done! ({evaluation_time:,.2f}s)")

    # If we do not compute the log probability, set it to None (not NaN)
    # This is to distinguish it from something going wrong in the computation
    # of the log probability.
    else:
        avg_logprob = None

    return avg_loss, avg_logprob

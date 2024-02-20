"""
Methods to train (or validate) a given model for one epoch.
"""

from typing import Any, TYPE_CHECKING, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fm4ar.utils.torchutils import check_for_nans, perform_scheduler_step
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
    stage_config: dict[str, Any],
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
    use_amp = stage_config.get("use_amp", False)
    gradient_clipping_config = stage_config.get("gradient_clipping", {})
    loss_kwargs = stage_config.get("loss_kwargs", {})

    # Check if we can use automatic mixed precision
    if use_amp and model.device == torch.device("cpu"):  # pragma: no cover
        raise RuntimeError("Don't use automatic mixed precision on CPU!")

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
    scaler = GradScaler(enabled=use_amp)  # type: ignore

    # Iterate over the batches
    batch: dict[str, torch.Tensor]
    for batch_idx, batch in enumerate(dataloader):

        loss_info.update_timer()

        # Move data to device
        theta, context = move_batch_to_device(batch, model.device)

        # Reset gradients
        model.optimizer.zero_grad(set_to_none=True)

        # No automatic mixed precision
        if not use_amp:

            loss = model.loss(theta=theta, context=context)
            check_for_nans(loss, "train loss")

            loss.backward()  # type: ignore

            if gradient_clipping_config:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.network.parameters(),
                    **gradient_clipping_config,
                )

            model.optimizer.step()

        # With automatic mixed precision (default)
        # This cannot be tested at the moment because it requires a GPU
        else:  # pragma: no cover

            # Note: Backward passes under autocast are not recommended
            with autocast():
                loss = model.loss(theta=theta, context=context, **loss_kwargs)
                check_for_nans(loss, "train loss")

            scaler.scale(loss).backward()  # type: ignore

            if gradient_clipping_config:
                scaler.unscale_(model.optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.network.parameters(),
                    **gradient_clipping_config,
                )

            scaler.step(model.optimizer)  # type: ignore
            scaler.update()  # type: ignore

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
    stage_config: dict[str, Any],
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

    # Check the `model` is an FMPE model. We cannot directly use `isinstance`
    # here, since this would create a circular import.
    is_fmpe_model = model.__class__.__name__ == "FMPEModel"

    # Set default value for `logprob_epochs`
    logprob_epochs = stage_config.get("logprob_epochs")
    if logprob_epochs is None:
        logprob_epochs = 10 if is_fmpe_model else 1

    # Get additional keyword arguments for loss function
    loss_kwargs = stage_config.get("loss_kwargs", {})

    # We don't need to compute gradients for the validation set
    with torch.no_grad():

        # Set up a LossInfo object to keep track of the loss and times
        loss_info = LossInfo(
            epoch=model.epoch,
            len_dataset=len(dataloader.dataset),  # type: ignore
            batch_size=dataloader.batch_size,  # type: ignore
            mode="Validate",
            print_freq=1,
        )

        # Store average logprob: We only compute this from the first batch,
        # because it otherwise takes extremely long for flow matching models.
        avg_logprob = None

        # Additional keyword arguments for log_prob_batch
        # Background: It seems that the time-inverse ODE required to compute
        # the log probability of the true parameter value is sometimes stiffer
        # than the "forward" ODE, so we need to use a different solver.
        # TODO: Check this again more thoroughly!
        log_prob_kwargs: dict[str, Any] = dict()
        if is_fmpe_model:
            log_prob_kwargs["tolerance"] = 1e-3
            log_prob_kwargs["method"] = "dopri8"

        # Iterate over the batches
        batch: dict[str, torch.Tensor]
        for batch_idx, batch in enumerate(dataloader):

            loss_info.update_timer()

            # Move data to device
            theta, context = move_batch_to_device(batch, model.device)

            # Compute validation loss
            loss = model.loss(theta=theta, context=context, **loss_kwargs)
            check_for_nans(loss, "validation loss")

            # Define maximum number of samples to use for log probability.
            # This is to limit the memory usage for batch sizes that cannot
            # be processed without AMP anymore.
            # TODO: Maybe there is a cleaner way of handling this?
            MAX_SAMPLES_FOR_LOGPROB = 1024

            # Compute log probability of true parameter values of first batch
            if (
                logprob_epochs > 0
                and (model.epoch - 1) % logprob_epochs == 0
                and batch_idx == 0
            ):
                logprob = model.log_prob_batch(
                    theta=theta[:MAX_SAMPLES_FOR_LOGPROB],
                    context={
                        key: value[:MAX_SAMPLES_FOR_LOGPROB]
                        for key, value in context.items()
                    },
                    **log_prob_kwargs,  # for FMPE: tolerance, method
                ).cpu()
                avg_logprob = float(logprob.mean().item())

            # Update loss for history and logging
            loss_info.update(loss.item(), len(theta))
            loss_info.print_info(batch_idx)

        # Return the average validation loss and log probability
        return loss_info.get_avg(), avg_logprob

"""
Base class which provides basic functionality for training and
inference, and from which all posterior models should inherit.
"""

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, Union
from warnings import catch_warnings, simplefilter

import pandas as pd
import torch
import wandb
from threadpoolctl import threadpool_limits
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from fm4ar.utils.torchutils import (
    check_for_nans,
    get_lr,
    get_optimizer_from_kwargs,
    get_scheduler_from_kwargs,
    perform_scheduler_step,
)
from fm4ar.utils.tracking import LossInfo, RuntimeLimits

if TYPE_CHECKING:
    from fm4ar.models.continuous.flow_matching import FlowMatching
    from fm4ar.models.discrete.normalizing_flow import NormalizingFlow


class Base:
    """
    Base class for posterior models.

    All posterior models (e.g., `NormalizingFlow`, `FlowMatching`, ...)
    should inherit from this class, which provides basic functionality
    for training and inference.
    """

    # Declare attributes with type hints (but without assigning values)
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau

    def __init__(
        self,
        experiment_dir: Path | None = None,
        file_path: Path | None = None,
        config: dict | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        load_training_info: bool = True,
    ) -> None:
        """
        Initialize a model for the posterior distribution.

        Args:
            experiment_dir: Path to the experiment directory. Required
                for operations like saving the model. If `None`, these
                operations will be skipped.
            file_path: Path to a checkpoint file. If given, the model
                will be loaded from this file.
            config: Experiment configuration. If given, the model will
                be initialized from these settings.
            device: Device to which the model should be moved.
            load_training_info: Whether to load training information
                (i.e., the state of the optimizer and LR scheduler)
                when loading the model from a checkpoint file.
        """

        # Store constructor arguments
        self.config = dict({} if config is None else config)
        self.device = torch.device(device)
        self.experiment_dir = experiment_dir

        # Initialize attributes
        self.epoch = 0
        self.optimizer_kwargs: dict | None = None
        self.model_kwargs: dict | None = None
        self.scheduler_kwargs: dict | None = None

        # Add dataframe that will keep track of the training history
        self.history: pd.DataFrame = pd.DataFrame()

        # Either load the model from a checkpoint file...
        if file_path is not None:
            self.load_model(
                file_path=file_path,
                load_training_info=load_training_info,
                device=device,
            )

        # ...or initialize it from the configuration
        else:
            self.initialize_model()
            self.model_to_device(device)

    @abstractmethod
    def initialize_model(self) -> None:
        """
        Initialize the model backbone.
        """

        raise NotImplementedError()

    @abstractmethod
    def sample_batch(
        self,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Sample a batch of data from the posterior model.
        """

        raise NotImplementedError()

    @abstractmethod
    def sample_and_log_prob_batch(
        self,
        context: torch.Tensor | None,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of data and log probs from the posterior model.
        """

        raise NotImplementedError()

    @abstractmethod
    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute the log probabilities of a batch of `theta`.
        """

        raise NotImplementedError()

    @abstractmethod
    def loss(
        self,
        theta: torch.Tensor,
        context: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss for a batch.
        """

        raise NotImplementedError()

    def model_to_device(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """
        Move model to `device`, and set `self.device` accordingly.
        """

        if device not in ("cpu", "cuda"):
            raise ValueError(f"Invalid device: `{device}`.")

        self.device = torch.device(device)
        self.model.to(self.device)

    def initialize_optimizer_and_scheduler(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler.
        """

        if self.optimizer_kwargs is not None:
            self.optimizer = get_optimizer_from_kwargs(
                model_parameters=self.model.parameters(),
                **self.optimizer_kwargs,
            )

        if self.scheduler_kwargs is not None:
            self.scheduler = get_scheduler_from_kwargs(
                optimizer=self.optimizer,
                **self.scheduler_kwargs,
            )

    def log_metrics(self, **kwargs: Any) -> None:
        """
        Add a row to the training history (locally and wandb).
        """

        # Convert kwargs to a row and append it to the history dataframe
        new_row = pd.DataFrame(kwargs, index=[0])

        if self.history.empty:
            self.history = new_row

        else:

            # Ignore a FutureWarning from pandas ("The behavior of DataFrame
            # concatenation with empty or all-NA entries is deprecated.") that
            # is caused by the fact that we are concatenating a dataframe with
            # a single row in which some columns may be NaN.
            with catch_warnings():
                simplefilter("ignore", category=FutureWarning)
                self.history = pd.concat(
                    objs=[self.history, new_row],
                    ignore_index=True,
                    axis=0,
                )

        # Save the history to disk as a backup (this is never read)
        if self.experiment_dir is not None:
            file_path = self.experiment_dir / "history.csv"
            self.history.to_csv(file_path, index=False)

        # Save the history to wandb
        if self.use_wandb:
            wandb.log(kwargs)

    def save_model(
        self,
        name: str = "latest",
        prefix: str = "model",
        save_training_info: bool = True,
        target_dir: Path | None = None,
    ) -> None:
        """
        Save the posterior model to disk.

        Args:
            prefix: The prefix for the model name (default: 'model').
            name: Model name (e.g., "latest" or "best").
            save_training_info: Whether to save training information
                that is required to continue training (i.e., the state
                dicts of the optimizer and LR scheduler).
            target_dir: Directory to which the model should be saved.
                Usually, this is the experiment directory, but it can
                also be a different directory (e.g., "snapshots").
        """

        # If no experiment directory is given, we don't save anything
        if self.experiment_dir is None:
            return

        # Collect all the data that we want to save
        data = {
            "config": self.config,
            "epoch": self.epoch,
            "history": self.history,
            "model_state_dict": self.model.state_dict(),
        }

        # Add optional data
        if save_training_info:
            data["optimizer_kwargs"] = self.optimizer_kwargs
            data["scheduler_kwargs"] = self.scheduler_kwargs
            if self.optimizer is not None:
                data["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save the data to disk
        if target_dir is None:
            target_dir = self.experiment_dir
        file_path = target_dir / f"{prefix}__{name}.pt"
        torch.save(obj=data, f=file_path)

    def load_model(
        self,
        file_path: Path,
        load_training_info: bool = True,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """
        Load a posterior model from disk.

        Args:
            file_path: Path to saved model.
            load_training_info: Whether to load information required to
                continue training, e.g., the optimizer state dict.
            device: Device to load the model on.
        """

        # Load data from disk and move everything to the correct device
        data = torch.load(file_path, map_location=device)

        # Load required data
        self.epoch = data["epoch"]
        self.config = data["config"]
        self.history = data["history"]

        # Initialize model and load state dict
        self.initialize_model()
        self.model.load_state_dict(data["model_state_dict"])
        self.model_to_device(device)

        # Set up optimizer and learning rate scheduler for resuming training
        if load_training_info:

            # Load optimizer and scheduler kwargs
            if "optimizer_kwargs" in data:
                self.optimizer_kwargs = data["optimizer_kwargs"]
            if "scheduler_kwargs" in data:
                self.scheduler_kwargs = data["scheduler_kwargs"]

            # Initialize optimizer and scheduler
            self.initialize_optimizer_and_scheduler()

            # Load optimizer and scheduler state dict
            if "optimizer_state_dict" in data:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            if "scheduler_state_dict" in data:
                self.scheduler.load_state_dict(data["scheduler_state_dict"])

    @property
    def checkpoint_epochs(self) -> int:
        return int(self.config["local"].get("checkpoint_epochs", 1))

    @property
    def use_wandb(self) -> bool:
        return self.config["local"].get("wandb") is not None

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        runtime_limits: RuntimeLimits,
        stage_config: dict[str, Any],
    ) -> bool:
        """
        Train the model until the runtime limits are exceeded.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            runtime_limits: RuntimeLimits object.
            stage_config: Configuration for the current training stage.

        Returns:
            True if we stopped because we reached the early stopping
            criterion, False otherwise.
        """

        # Run for as long as the runtime limits allow
        while not runtime_limits.limits_exceeded(self.epoch):

            self.epoch += 1

            # Run one epoch of training and testing
            lr = get_lr(self.optimizer)
            with threadpool_limits(limits=1, user_api="blas"):

                # Train for one epoch and measure the time
                print(f"\nStart training epoch {self.epoch} with lr {lr}:")
                train_start = time.time()
                train_loss = train_epoch(
                    pm=self,
                    dataloader=train_loader,
                    stage_config=stage_config,
                )
                train_time = time.time() - train_start
                print(f"Done! This took {train_time:,.2f} seconds.\n")

                # Run on the test set and measure the time
                print(f"Start testing epoch {self.epoch}:")
                test_start = time.time()
                test_loss, test_logprob = test_epoch(
                    pm=self,
                    dataloader=test_loader,
                    stage_config=stage_config,
                )
                test_time = time.time() - test_start
                print(f"Done! This took {test_time:,.2f} seconds.\n")

            # Take a step with the learning rate scheduler after each epoch
            perform_scheduler_step(
                scheduler=self.scheduler,
                loss=test_loss,
                end_of="epoch",
                on_lower=self.save_snapshot,  # only for ReduceLROnPlateau
            )

            # Log relevant metrics (both locally and on wandb)
            print("Logging metrics...", end=" ")
            self.log_metrics(
                epoch=self.epoch,
                learning_rate=lr[0],
                train_loss=train_loss,
                test_loss=test_loss,
                train_time=train_time,
                test_time=test_time,
                test_logprob=test_logprob,
            )
            print("Done!")

            # Save the latest model
            print("Saving latest model...", end=" ")
            self.save_model()
            print("Done!")

            # Check if we should stop early
            if self.stop_early(patience=stage_config.get("early_stopping")):
                print("Early stopping criterion reached, ending training!")
                return True

            # Save the best model if the test loss has improved
            self.save_best_model(test_loss=test_loss)
            print()

        return False

    def save_best_model(self, test_loss: float) -> None:
        """
        Check if the current model is the best one, and if yes, save it.
        """

        # Note: This should not be needed because this function should only
        # be called after `.log()` has been called at least once, but...
        try:
            best_loss = float(self.history["test_loss"].min())
        except KeyError:
            best_loss = float("inf")

        # Note: "<=" (instead of "<") is important here!
        if test_loss <= best_loss:
            print("Saving best model...", end=" ")
            self.save_model(name="best", save_training_info=False)
            print("Done!")

    def save_snapshot(self) -> None:
        """
        Save a snapshot of the model.
        """

        # If no experiment directory is given, we don't save anything
        if self.experiment_dir is None:
            return

        # Create the snapshots directory if it doesn't exist yet
        snapshots_dir = self.experiment_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        print("Saving snapshot...", end=" ")
        self.save_model(
            prefix="snapshot",
            name=f"{self.epoch:04d}",
            save_training_info=False,
            target_dir=snapshots_dir,
        )
        print("Done!")

    def stop_early(self, patience: int | None) -> bool:
        """
        Check if we should stop early: If the test loss has not improved
        for `patience` epochs, we stop the training.
        """

        if patience is None:
            return False

        min_idx = int(self.history["test_loss"].values.argmin())
        last_idx = len(self.history)

        return bool(last_idx - min_idx > patience)


def train_epoch(
    pm: Base,
    dataloader: DataLoader,
    stage_config: dict[str, Any],
) -> float:
    """
    Train the posterior model for one epoch.

    Args:
        pm: Posterior model to train.
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
    if use_amp and pm.device == torch.device("cpu"):
        raise RuntimeError("Don't use automatic mixed precision on CPU!")

    # Ensure that the model is in training mode
    pm.model.train()

    # Set up a LossInfo object to keep track of the loss and times
    loss_info = LossInfo(
        epoch=pm.epoch,
        len_dataset=len(dataloader.dataset),  # type: ignore
        batch_size=int(dataloader.batch_size),  # type: ignore
        mode="Train",
        print_freq=1,
    )

    # Create scaler for automatic mixed precision
    scaler = GradScaler(enabled=use_amp)  # type: ignore

    # Iterate over the batches
    for batch_idx, (theta, context) in enumerate(dataloader):

        # Move data to device
        theta = theta.to(pm.device, non_blocking=True)
        if context is not None:
            context = context.to(pm.device, non_blocking=True)

        loss_info.update_timer()
        pm.optimizer.zero_grad()

        # No automatic mixed precision
        if not use_amp:

            loss = pm.loss(theta=theta, context=context)
            check_for_nans(loss, "train loss")

            loss.backward()  # type: ignore

            if gradient_clipping_config:
                torch.nn.utils.clip_grad_norm_(
                    parameters=pm.model.parameters(),
                    **gradient_clipping_config,
                )

            pm.optimizer.step()

        # With automatic mixed precision (default)
        else:

            # Note: Backward passes under autocast are not recommended
            with autocast():
                loss = pm.loss(theta=theta, context=context, **loss_kwargs)
                check_for_nans(loss, "train loss")

            scaler.scale(loss).backward()  # type: ignore

            if gradient_clipping_config:
                scaler.unscale_(pm.optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(
                    parameters=pm.model.parameters(),
                    **gradient_clipping_config,
                )

            scaler.step(pm.optimizer)  # type: ignore
            scaler.update()  # type: ignore

        # Take a step with the learning rate scheduler after each batch.
        # This is required, e.g., for the OneCycleLR scheduler.
        perform_scheduler_step(
            scheduler=pm.scheduler,
            loss=None,
            end_of="batch",
        )

        # Update loss for history and logging
        loss_info.update(loss.detach().item(), n=len(theta))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch(
    pm: Union[Base, "FlowMatching", "NormalizingFlow"],
    dataloader: DataLoader,
    stage_config: dict[str, Any],
) -> tuple[float, float | None]:
    """
    Test the posterior model on the test set for one epoch.

    Args:
        pm: Posterior model to test.
        dataloader: Dataloader for test data.
        stage_config: Configuration for the current training stage.

    Returns:
        A 2-tuple, consisting of:
        (1) The average test loss over the epoch.
        (2) The average log probability of the true parameter values.
    """

    # TODO: Maybe we also want to use AMP here to speed up the logprob stuff?

    pm.model.eval()

    # Determine the type of the posterior model
    # Note: We can't directly check if `pm` is a `FlowMatching` instance,
    # since this would create a circular import...
    if hasattr(pm, "evaluate_vectorfield"):
        model_type = "FlowMatching"
    else:
        model_type = "NormalizingFlow"

    # Set default value for `logprob_epochs`
    logprob_epochs = stage_config.get("logprob_epochs")
    if logprob_epochs is None:
        logprob_epochs = 10 if model_type == "FlowMatching" else 1

    # Get additional keyword arguments for loss function
    loss_kwargs = stage_config.get("loss_kwargs", {})

    # We don't need to compute gradients for the test set
    with torch.no_grad():

        # Set up a LossInfo object to keep track of the loss and times
        loss_info = LossInfo(
            epoch=pm.epoch,
            len_dataset=len(dataloader.dataset),  # type: ignore
            batch_size=dataloader.batch_size,  # type: ignore
            mode="Test",
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
        if model_type == "FlowMatching":
            log_prob_kwargs["tolerance"] = 1e-3
            log_prob_kwargs["method"] = "dopri8"

        # Iterate over the batches
        for batch_idx, (theta, context) in enumerate(dataloader):
            loss_info.update_timer()

            # Move data to device
            theta = theta.to(pm.device, non_blocking=True)
            if context is not None:
                context = context.to(pm.device, non_blocking=True)

            # Compute test loss
            loss = pm.loss(theta=theta, context=context, **loss_kwargs)
            check_for_nans(loss, "test loss")

            # Define maximum number of samples to use for log probability.
            # This is to limit the memory usage for batch sizes that cannot
            # be processed without AMP anymore.
            # TODO: Maybe there is a cleaner way of handling this?
            MAX_SAMPLES_FOR_LOGPROB = 1024

            # Compute log probability of true parameter values of first batch
            if (
                logprob_epochs > 0
                and (pm.epoch - 1) % logprob_epochs == 0
                and batch_idx == 0
            ):
                logprob = pm.log_prob_batch(
                    theta=theta[:MAX_SAMPLES_FOR_LOGPROB],
                    context=context[:MAX_SAMPLES_FOR_LOGPROB],
                    **log_prob_kwargs
                ).cpu()
                avg_logprob = float(logprob.mean().item())

            # Update loss for history and logging
            loss_info.update(loss.item(), len(theta))
            loss_info.print_info(batch_idx)

        # Return the average test loss and log probability
        return loss_info.get_avg(), avg_logprob

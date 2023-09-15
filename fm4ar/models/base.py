"""
Base class which provides basic functionality for training and
inference, and from which all posterior models should inherit.
"""

import shutil
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, Union

import torch
import wandb
from threadpoolctl import threadpool_limits
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from fm4ar.utils.torchutils import (
    get_lr,
    get_optimizer_from_kwargs,
    get_scheduler_from_kwargs,
    perform_scheduler_step,
)
from fm4ar.utils.tracking import (
    EarlyStopping,
    LossInfo,
    RuntimeLimits,
    write_history,
)

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
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau

    def __init__(
        self,
        file_path: Path | None = None,
        config: dict | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        load_training_info: bool = True,
    ) -> None:
        """
        Initialize a model for the posterior distribution.

        Args:
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

        # Initialize attributes
        self.epoch = 0
        self.optimizer_kwargs: dict | None = None
        self.network_kwargs: dict | None = None
        self.scheduler_kwargs: dict | None = None

        # Either load the model from a checkpoint file...
        if file_path is not None:
            self.load_model(
                file_path=file_path,
                load_training_info=load_training_info,
                device=device,
            )

        # ...or initialize it from the configuration
        else:
            self.initialize_network()
            self.network_to_device(device)

    @abstractmethod
    def initialize_network(self) -> None:
        """
        Initialize the network backbone for the posterior model.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_batch(
        self,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a batch of data from the posterior model.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_and_log_prob_batch(
        self,
        *context_data: torch.Tensor,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of data and log probs from the posterior model.
        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob_batch(
        self,
        data: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log probabilities of a batch of `data`.
        """
        raise NotImplementedError()

    @abstractmethod
    def loss(
        self,
        data: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for a batch of `data`.
        """
        raise NotImplementedError()

    def network_to_device(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """
        Move model to `device`, and set `self.device` accordingly.
        """

        if device not in ("cpu", "cuda"):
            raise ValueError(f"Invalid device: `{device}`.")

        self.device = torch.device(device)
        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler.
        """

        if self.optimizer_kwargs is not None:
            self.optimizer = get_optimizer_from_kwargs(
                model_parameters=self.network.parameters(),
                **self.optimizer_kwargs,
            )

        if self.scheduler_kwargs is not None:
            self.scheduler = get_scheduler_from_kwargs(
                optimizer=self.optimizer,
                **self.scheduler_kwargs,
            )

    def save_model(
        self,
        experiment_dir: Path,
        name: str = "latest",
        prefix: str = "model",
        checkpoint_epochs: int | None = None,
        save_training_info: bool = True,
    ) -> None:
        """
        Save the posterior model to disk.

        Args:
            experiment_dir: The directory to save the model in.
            prefix: The prefix for the model name (default: 'model').
            name: Model name (e.g., "latest" or "best").
            checkpoint_epochs: The number of epochs between two
                consecutive
                model checkpoints.
            save_training_info: Whether to save training information
                that is required to continue training (i.e., the state
                dicts of the optimizer and LR scheduler).
        """

        # Collect all the data that we want to save
        data = {
            "config": self.config,
            "epoch": self.epoch,
            "model_state_dict": self.network.state_dict(),
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
        file_path = experiment_dir / f"{prefix}__{name}.pt"
        torch.save(obj=data, f=file_path)

        # If no checkpoint is requested, we are done here
        if checkpoint_epochs is None or self.epoch % checkpoint_epochs != 0:
            return

        # Otherwise, ensure that the checkpoints directory exists
        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        # Create a backup of the model (in case something gets corrupted)
        src = file_path
        dst = checkpoints_dir / f"{prefix}__{self.epoch:04d}.pt"
        shutil.copyfile(src, dst)

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

        # Initialize network and load state dict
        self.initialize_network()
        self.network.load_state_dict(data["model_state_dict"])
        self.network_to_device(device)

        # Set up optimizer and learning rate scheduler for resuming training
        if load_training_info:
            self.network.train()

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

        else:
            self.network.eval()

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        experiment_dir: Path,
        runtime_limits: RuntimeLimits,
        checkpoint_epochs: int | None = None,
        use_wandb: bool = False,
        test_only: bool = False,
        early_stopping_config: dict | None = None,
        gradient_clipping_config: dict | None = None,
        use_amp: bool = True,
        logprob_epochs: int | None = None,
    ) -> None:
        """
        Train the model until the runtime limits are exceeded.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            experiment_dir: Path to the experiment directory.
            runtime_limits: RuntimeLimits object.
            checkpoint_epochs: Number of epochs between checkpoints.
            use_wandb: Whether to use wandb for logging.
            early_stopping_config: Configuration for EarlyStopping.
            test_only: Whether to only evaluate the model on the test
                set (i.e., skip training).
            gradient_clipping_config: Configuration for gradient
                clipping (will be passed to `nn.utils.clip_grad_norm_`).
            use_amp: Whether to use automatic mixed precision.
            logprob_epochs: Evaluate the log probability of the true
                parameter values every `logprob_epochs` epochs. (See
                `test_epoch()` for details.)
        """

        # ---------------------------------------------------------------------
        # In test mode, we only want to evaluate the model on the test set
        # ---------------------------------------------------------------------

        if test_only:
            test_loss, test_logprob = test_epoch(
                pm=self,
                dataloader=test_loader,
                epoch=1,
                logprob_epochs=1,
            )
            print(f"Mean test loss:    {test_loss:.3f}")
            print(f"Mean test logprob: {test_logprob:.3f}")
            return

        # ---------------------------------------------------------------------
        # Otherwise, we actually train the model
        # ---------------------------------------------------------------------

        # Set up the EarlyStopping tracker
        if early_stopping_config is not None and early_stopping_config:
            early_stopping = EarlyStopping(**early_stopping_config)
        else:
            early_stopping = None

        # Run for as long as the runtime limits allow
        while not runtime_limits.limits_exceeded(self.epoch):
            self.epoch += 1

            # Run one epoch of training and testing
            lr = get_lr(self.optimizer)
            with threadpool_limits(limits=1, user_api="blas"):
                # Train for one epoch and measure the time
                print(f"\nStart training epoch {self.epoch} with lr {lr}:\n")
                time_start = time.time()
                train_loss = train_epoch(
                    pm=self,
                    dataloader=train_loader,
                    gradient_clipping_config=gradient_clipping_config,
                    use_amp=use_amp,
                )
                train_time = time.time() - time_start
                print(f"\nDone! This took {train_time:,.2f} seconds.\n")

                # Run on the test set and measure the time
                print(f"\nStart testing epoch {self.epoch}:\n")
                time_start = time.time()
                test_loss, test_logprob = test_epoch(
                    pm=self,
                    dataloader=test_loader,
                    epoch=self.epoch,
                    logprob_epochs=logprob_epochs,
                )
                test_time = time.time() - time_start
                print(f"\nDone! This took {test_time:,.2f} seconds.\n")

            # Take a step with the learning rate scheduler after each epoch
            perform_scheduler_step(
                scheduler=self.scheduler,
                loss=test_loss,
                end_of="epoch",
            )

            # Write the training history to a log file
            write_history(
                experiment_dir=experiment_dir,
                epoch=self.epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                learning_rates=lr,
            )

            # Save the latest model to a checkpoint file
            self.save_model(
                experiment_dir=experiment_dir,
                checkpoint_epochs=checkpoint_epochs,
            )

            # Log the results for this epoch to Weights & Biases
            if use_wandb:
                wandb.define_metric("epoch")
                wandb.define_metric("*", step_metric="epoch")
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "learning_rate": lr[0],
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "train_time": train_time,
                        "test_time": test_time,
                        "test_logprob": test_logprob,
                    }
                )

            # Check if we should stop early
            if early_stopping is not None:
                # Check if the current model is the best one yet
                # TODO: This does not work across multiple jobs / restarts;
                #      we probably need to read the history file for this...
                is_best_model = early_stopping(test_loss)
                if is_best_model:
                    self.save_model(
                        experiment_dir=experiment_dir,
                        name="best",
                        save_training_info=False,
                    )

                # Check if we should stop early
                if early_stopping.early_stop:
                    print("Early stopping criterion reached!")
                    break

            print(f"Finished training epoch {self.epoch}.\n")


def train_epoch(
    pm: Base,
    dataloader: DataLoader,
    gradient_clipping_config: dict[str, Any] | None = None,
    use_amp: bool = True,
    verbose: bool = True,
) -> float:
    """
    Train the posterior model for one epoch.

    Args:
        pm: Posterior model to train.
        dataloader: Dataloader for training data.

        gradient_clipping_config: Configuration for gradient clipping.
            If `None` or `{}`, no gradient clipping is performed.
            Otherwise, it is expected to be a dictionary that can be
            passed to `torch.nn.utils.clip_grad_norm_`. It must at
            least contain the key "max_norm".
        use_amp: Whether to use automatic mixed precision.
        verbose: Whether to print progress information.

    Returns:
        Average loss over the epoch.
    """

    # Check if we can use automatic mixed precision
    if use_amp and pm.device == torch.device("cpu"):
        raise RuntimeError("Don't use automatic mixed precision on CPU!")

    # Ensure that the model is in training mode
    pm.network.train()

    # Set up a LossInfo object to keep track of the loss and times
    loss_info = LossInfo(
        epoch=pm.epoch,
        len_dataset=len(dataloader.dataset),  # type: ignore
        batch_size=int(dataloader.batch_size),  # type: ignore
        mode="Train",
        print_freq=1,
    )

    # Create scaler for automatic mixed precision
    scaler = GradScaler()  # type: ignore

    # Iterate over the batches
    for batch_idx, data in enumerate(dataloader):

        # Move data to device
        data = [d.to(pm.device, non_blocking=True) for d in data]

        loss_info.update_timer()
        pm.optimizer.zero_grad()

        # No automatic mixed precision
        if not use_amp:

            loss = pm.loss(data[0], *data[1:])
            loss.backward()  # type: ignore

            if (
                gradient_clipping_config is not None
                and gradient_clipping_config
            ):
                torch.nn.utils.clip_grad_norm_(
                    parameters=pm.network.parameters(),
                    **gradient_clipping_config,
                )

            pm.optimizer.step()

        # With automatic mixed precision (default)
        else:

            with autocast():
                loss = pm.loss(data[0], *data[1:])
            scaler.scale(loss).backward()  # type: ignore

            if (
                gradient_clipping_config is not None
                and gradient_clipping_config
            ):
                scaler.unscale_(pm.optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(
                    parameters=pm.network.parameters(),
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
        loss_info.update(loss.detach().item(), n=len(data[0]))

        if verbose:
            loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch(
    pm: Union[Base, "FlowMatching", "NormalizingFlow"],
    dataloader: DataLoader,
    epoch: int,
    logprob_epochs: int | None = None,
) -> tuple[float, float | None]:
    """
    Test the posterior model on the test set for one epoch.

    Args:
        pm: Posterior model to test.
        dataloader: Dataloader for test data.
        epoch: Current epoch (count starts at 1).
        logprob_epochs: Evaluate the log probability of the true
            parameter values every `logprob_epochs` epochs.
            If `None`, a default value is chosen based on `pm`: For
            `NormalizingFlow` models, the log probability is evaluated
            every epochs, while for `FlowMatching` models, it is only
            evaluated every 10 epochs.

    Returns:
        A 2-tuple, consisting of:
        (1) The average test loss over the epoch.
        (2) The average log probability of the true parameter values.
    """

    # TODO: Maybe we also want to use AMP here to speed up the logprob stuff?

    pm.network.eval()

    # Determine the type of the posterior model
    # Note: We can't directly check if `pm` is a `FlowMatching` instance,
    # since this would create a circular import...
    if hasattr(pm, "evaluate_vectorfield"):
        model_type = "FlowMatching"
    else:
        model_type = "NormalizingFlow"

    # Set default value for `logprob_epochs`
    if logprob_epochs is None:
        logprob_epochs = 10 if model_type == "FlowMatching" else 1

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

        # Store all log probabilities in a list
        list_of_logprob: list[torch.Tensor] = []

        # Additional keyword arguments for log_prob_batch
        # Background: It seems that the time-inverse ODE required to compute
        # the log probability of the true parameter value is sometimes stiffer
        # than the "forward" ODE, so we need to use a different solver.
        # TODO: Check this more thoroughly!
        log_prob_kwargs: dict[str, Any] = dict()
        if model_type == "FlowMatching":
            log_prob_kwargs["tolerance"] = 1e-3
            log_prob_kwargs["method"] = "dopri8"

        # Iterate over the batches
        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()

            # Move data to device
            data = [d.to(pm.device, non_blocking=True) for d in data]

            # Compute test loss
            loss = pm.loss(data[0], *data[1:])

            # Compute log probability of true parameter values
            if logprob_epochs is not None and epoch % logprob_epochs == 0:
                logprob = pm.log_prob_batch(
                    data[0],
                    *data[1:],
                    **log_prob_kwargs

                ).cpu()
                list_of_logprob.append(logprob)

            # Update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        # Return the average test loss and log probability
        if logprob_epochs is not None and epoch % logprob_epochs == 0:
            return loss_info.get_avg(), None
        avg_logprob = float(torch.cat(list_of_logprob).mean().item())
        return loss_info.get_avg(), avg_logprob

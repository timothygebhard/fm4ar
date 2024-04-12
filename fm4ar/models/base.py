"""
Base class which provides basic functionality for training and
inference, and from which all posterior models should inherit.
"""

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal
from warnings import catch_warnings, simplefilter, warn

import pandas as pd
import torch
import wandb
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from fm4ar.training.stages import ExitStatus, StageConfig
from fm4ar.training.train_validate import validate_epoch, train_epoch
from fm4ar.torchutils.early_stopping import early_stopping_criterion_reached
from fm4ar.torchutils.general import resolve_device
from fm4ar.torchutils.optimizers import (
    OptimizerConfig,
    get_optimizer_from_config,
    get_lr,
)
from fm4ar.torchutils.schedulers import (
    Scheduler,
    SchedulerConfig,
    get_scheduler_from_config,
    perform_scheduler_step,
)
from fm4ar.utils.tracking import RuntimeLimits


class Base:
    """
    Base class for all models. Both the `FMPEModel` and the `NPEModel`
    inherit from this class, which provides basic functionality for
    training, loading / saving, etc.
    """

    # Declare attributes with type hints (but without assigning values)
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Scheduler

    def __init__(
        self,
        experiment_dir: Path | None = None,
        file_path: Path | None = None,
        config: dict | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        load_training_info: bool = True,
        random_seed: int | None = 42,
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
            random_seed: Random seed used for model initialization.
        """

        # Store constructor arguments
        self.config = dict({} if config is None else config)
        self.device = resolve_device(device)
        self.experiment_dir = experiment_dir
        self.random_seed = random_seed

        # Initialize attributes
        self.epoch: int = 0  # global epoch (across all stages)
        self.stage_name: str | None = None  # name of current training stage
        self.stage_epoch: int = 0  # epoch within current training stage
        self.model_config: dict | None = None
        self.optimizer_config: OptimizerConfig | None = None
        self.scheduler_config: SchedulerConfig | None = None

        # Add dataframe that will keep track of the training history
        self.history: pd.DataFrame = pd.DataFrame()

        # Either load the network from a checkpoint file...
        if file_path is not None:
            self.load_model(
                file_path=file_path,
                load_training_info=load_training_info,
            )

        # ...or initialize it from the configuration
        else:
            self.initialize_network()
            self.network_to_device()

    @abstractmethod
    def initialize_network(self) -> None:
        """
        Initialize the networks that are the core of the model.
        """

        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def sample_batch(
        self,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Sample a batch of data from the posterior model.
        """

        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def sample_and_log_prob_batch(
        self,
        context: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of data and log probs from the posterior model.
        """

        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the log probabilities of a batch of `theta`.
        """

        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def loss(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss for a batch.
        """

        raise NotImplementedError()  # pragma: no cover

    def network_to_device(self) -> None:
        """
        Move network to `self.device`.
        """

        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler.
        """

        if self.optimizer_config is not None:
            self.optimizer = get_optimizer_from_config(
                model_parameters=self.network.parameters(),
                optimizer_config=self.optimizer_config,
            )

        if self.scheduler_config is not None:
            self.scheduler = get_scheduler_from_config(
                optimizer=self.optimizer,
                scheduler_config=self.scheduler_config,
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

            # We could create a local copy of the history dataframe here, but
            # since we are currently not using it, this might just waste I/O.

        # Save the history to wandb
        if self.use_wandb:  # pragma: no cover
            wandb.log(kwargs)

    def save_model(
        self,
        name: str = "latest",
        prefix: str = "model",
        save_training_info: bool = True,
        target_dir: Path | None = None,
    ) -> Path | None:
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

        Returns:
            Path to the saved model.
        """

        # If no directory is given, we don't save anything
        if self.experiment_dir is None and target_dir is None:
            warn(
                UserWarning(
                    "save_model() was called, but no directory was specified!"
                )
            )
            return None

        # Collect all the data that we want to save
        data = {
            "config": self.config,
            "epoch": self.epoch,
            "stage_name": self.stage_name,
            "stage_epoch": self.stage_epoch,
            "history": self.history,
            "network_state_dict": self.network.state_dict(),
        }

        # Add optional data
        if save_training_info:
            data["optimizer_config"] = self.optimizer_config
            data["scheduler_config"] = self.scheduler_config
            if self.optimizer is not None:
                data["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save the data to disk
        if target_dir is None:
            target_dir = self.experiment_dir
        assert target_dir is not None  # mypy is a bit dumb
        file_path = target_dir / f"{prefix}__{name}.pt"
        torch.save(obj=data, f=file_path)

        return file_path

    def load_model(
        self,
        file_path: Path,
        load_training_info: bool = True,
    ) -> None:
        """
        Load a posterior model (`FMPEModel` or `NPEModel`) from disk.

        Args:
            file_path: Path to saved model.
            load_training_info: Whether to load information required to
                continue training, e.g., the optimizer state dict.
        """

        # Load data from disk and move everything to the correct device
        data = torch.load(file_path, map_location=self.device)

        # Load some required metadata
        self.epoch = data["epoch"]
        self.stage_name = data["stage_name"]
        self.stage_epoch = data["stage_epoch"]
        self.config = data["config"]
        self.history = data["history"]

        # Initialize network, load state dict, and move to device
        self.initialize_network()
        self.network.load_state_dict(data["network_state_dict"])
        self.network_to_device()

        # Set up optimizer and learning rate scheduler for resuming training
        if load_training_info:

            # Load optimizer and scheduler kwargs
            if "optimizer_config" in data:
                self.optimizer_config = data["optimizer_config"]
            if "scheduler_config" in data:
                self.scheduler_config = data["scheduler_config"]

            # Initialize optimizer and scheduler
            self.initialize_optimizer_and_scheduler()

            # Load optimizer and scheduler state dict
            if "optimizer_state_dict" in data:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            if "scheduler_state_dict" in data:
                self.scheduler.load_state_dict(data["scheduler_state_dict"])

    @property
    def use_wandb(self) -> bool:
        return self.config["local"].get("wandb") is not None

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        runtime_limits: RuntimeLimits,
        stage_config: StageConfig,
    ) -> ExitStatus:
        """
        Train the model until the runtime limits are exceeded.

        Args:
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data.
            runtime_limits: RuntimeLimits object.
            stage_config: Configuration for the current training stage.

        Returns:
            ExitStatus indicating the reason for why this function
            stopped (completed, early stopped, or runtime exceeded).
        """

        # Run for as long as the runtime limits allow
        while not runtime_limits.limits_exceeded(self.epoch):

            # Increase epoch counters
            self.epoch += 1
            self.stage_epoch += 1

            # Run one epoch of training and testing
            lr = get_lr(self.optimizer)
            with threadpool_limits(limits=1, user_api="blas"):

                # Train for one epoch and measure the time
                print(f"\nStart training epoch {self.epoch} with lr {lr}:")
                train_start = time.time()
                train_loss = train_epoch(
                    model=self,
                    dataloader=train_loader,
                    stage_config=stage_config,
                )
                train_time = time.time() - train_start
                print(f"Done! This took {train_time:,.2f} seconds.\n")

                # Run on the validation set and measure the time
                print(f"Start validation epoch {self.epoch}:")
                test_start = time.time()
                test_loss, test_logprob = validate_epoch(
                    model=self,
                    dataloader=valid_loader,
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
            if early_stopping_criterion_reached(
                loss_history=self.history["test_loss"].values,
                stage_epoch=self.stage_epoch,
                early_stopping_config=stage_config.early_stopping,
            ):  # pragma: no cover
                print("Early stopping criterion reached, ending training!")
                return ExitStatus.EARLY_STOPPED

            # Save the best model if the test loss has improved
            self.save_best_model(test_loss=test_loss)
            print()

        return (
            ExitStatus.COMPLETED if not runtime_limits.max_runtime_exceeded()
            else ExitStatus.MAX_RUNTIME_EXCEEDED
        )

    def save_best_model(self, test_loss: float) -> None:
        """
        Check if the current model is the best one, and if yes, save it.
        """

        # Note: This should not be needed because this function should only
        # be called after `.log()` has been called at least once, but...
        try:
            best_loss = float(self.history["test_loss"].min())
        except KeyError:  # pragma: no cover
            best_loss = float("inf")

        # Note: "<=" (instead of "<") is important here!
        if test_loss <= best_loss:
            print("Saving best model...", end=" ")
            self.save_model(name="best", save_training_info=False)
            print("Done!")

    def save_snapshot(self) -> Path | None:
        """
        Save a snapshot of the model.
        """

        # If no experiment directory is given, we don't save anything
        if self.experiment_dir is None:  # pragma: no cover
            return None

        # Create the snapshots directory if it doesn't exist yet
        snapshots_dir = self.experiment_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        print("Saving snapshot...", end=" ")
        file_path = self.save_model(
            prefix="snapshot",
            name=f"{self.epoch:04d}",
            save_training_info=True,
            target_dir=snapshots_dir,
        )
        print("Done!")

        return file_path

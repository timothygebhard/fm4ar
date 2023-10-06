"""
Various classes for tracking, e.g., the average loss, or the runtime.
"""

import time
from typing import Literal


class AvgTracker:
    """
    Tracks the average of a value over time (e.g., runtime).
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.N = 0
        self.last = float("nan")

    def update(self, last: float, n: int = 1) -> None:
        self.sum += last
        self.N += n
        self.last = last

    def get_avg(self) -> float:
        return self.sum / self.N if self.N > 0 else float("nan")

    def get_last(self) -> float:
        return self.last


class LossInfo:
    """
    Keep track of the loss and computation times for a single epoch.
    """

    def __init__(
        self,
        epoch: int,
        len_dataset: int,
        batch_size: int,
        mode: str = "Train",
        print_freq: int = 8,
    ) -> None:
        """
        Initialize new LossInfo instance.

        Args:
            epoch: Current epoch (for print statements).
            len_dataset: Length of the dataset (for print statements).
            batch_size: Batch size (for print statements).
            mode: Mode (for print statements).
            print_freq: Print frequency (print every `N` batches).
        """

        # Data for print statements
        self.epoch = epoch
        self.len_dataset = len_dataset
        self.batch_size = batch_size
        self.mode = mode
        self.print_freq = print_freq

        # Track loss
        self.loss_tracker = AvgTracker()
        self.loss = float("nan")

        # Track computation times
        self.times = {
            "dataloader": AvgTracker(),
            "model": AvgTracker(),
        }
        self.t = time.time()

    def update_timer(
        self,
        timer_mode: Literal["dataloader", "model"] = "dataloader",
    ) -> None:
        """
        Update the timer (either for the dataloader or the model).
        """
        self.times[timer_mode].update(time.time() - self.t)
        self.t = time.time()

    def update(self, loss: float, n: int) -> None:
        """
        Update the loss and the timer.
        """
        self.loss = loss
        self.loss_tracker.update(loss * n, n)
        self.update_timer(timer_mode="model")

    def get_avg(self) -> float:
        """
        Return the average loss.
        """
        return self.loss_tracker.get_avg()

    def print_info(self, batch_idx: int) -> None:
        """
        Print progress, loss, and computation times.
        """

        # Print only every `print_freq` batches
        if batch_idx % self.print_freq != 0:
            return

        # Print progress (epoch, batch, percentage)
        processed = str(min(batch_idx * self.batch_size, self.len_dataset))
        processed = f"{processed}".rjust(len(f"{self.len_dataset}"))
        percentage = 100.0 * batch_idx * self.batch_size / self.len_dataset
        print(
            (
                f"[{self.mode}]  Epoch: {self.epoch:3d} "
                f"[{processed}/{self.len_dataset} = {percentage:>5.1f}%]"
            ),
            end="    ",
        )

        # Print loss (current and average)
        print(f"Loss = {self.loss:.3f} ({self.get_avg():.3f})", end="    ")

        # Print dataloader times (last and average) in milliseconds
        td, td_avg = (
            1_000 * self.times["dataloader"].get_last(),
            1_000 * self.times["dataloader"].get_avg(),
        )
        print(f"t_data = {td:.1f} ms ({td_avg:.1f} ms)", end="    ")

        # Print model times (last and average) in milliseconds
        tn, tn_avg = (
            1_000 * self.times["model"].get_last(),
            1_000 * self.times["model"].get_avg(),
        )
        print(f"t_model = {tn:.1f} ms ({tn_avg:.1f} ms)")


class RuntimeLimits:
    """
    Keeps track of the runtime limits (time and / or number of epochs)
    """

    def __init__(
        self,
        max_time_per_run: float | None = None,
        max_epochs_per_run: int | None = None,
        max_epochs_total: int | None = None,
        epoch_start: int = 0,
    ) -> None:
        """
        Initialize new `RuntimeLimits` object.

        Args:
            max_time_per_run: Maximum time for run, in seconds. This is
                a soft limit (i.e., the run will only be stopped after
                a full epoch).
            max_epochs_per_run: Maximum number of epochs for run.
            max_epochs_total: Maximum total number of training epochs.
            epoch_start: Start epoch of current run.
        """

        self.max_time_per_run = max_time_per_run
        self.max_epochs_per_run = max_epochs_per_run
        self.max_epochs_total = max_epochs_total
        self.epoch_start = epoch_start
        self.time_start = time.time()

    def limits_exceeded(self, epoch: int) -> bool:
        """
        Check whether any of the runtime limits are exceeded.
        """

        # Check time limit for run
        if self.max_time_per_run is not None:
            if time.time() - self.time_start >= self.max_time_per_run:
                print(f"Time limit of {self.max_time_per_run} s exceeded!")
                return True

        # Check epoch limit for run
        if self.max_epochs_per_run is not None:
            if epoch - self.epoch_start >= self.max_epochs_per_run:
                print(f"Run epoch limit of {self.max_epochs_per_run} reached!")
                return True

        # Check total epoch limit
        if self.max_epochs_total is not None:
            if epoch is None:
                raise ValueError("`epoch` argument is required!")
            if epoch >= self.max_epochs_total:
                print(f"Total epoch limit of {self.max_epochs_total} reached!")
                return True

        return False

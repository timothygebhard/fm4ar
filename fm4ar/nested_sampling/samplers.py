"""
Define abstractions for the different nested sampling implementations.
"""

import contextlib
import json
import time
import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Type

import dill
import multiprocess
import numpy as np

from fm4ar.utils.multiproc import get_number_of_available_cores
from fm4ar.utils.timeout import TimeoutException, timelimit


class Sampler(ABC):
    """
    Abstract base class for nested sampling samplers.
    """

    start_time: float
    complete: bool

    def __init__(
        self,
        run_dir: Path,
        prior_transform: Callable[[np.ndarray], np.ndarray],
        log_likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        inferred_parameters: list[str],
        random_seed: int = 42,
        sampler_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize the class instance.

        Args:
            run_dir: Path to the directory where the sampler should save
                its output.
            prior_transform: A function that transforms a sample from
                the `n_dim`-dimensional unit cube to the prior space.
            log_likelihood: A function that computes the log-likelihood
                of a given sample from the prior space.
            n_dim: The number of dimensions of the parameter space.
            n_livepoints: The number of live points to use in the
                nested sampling algorithm.
            inferred_parameters: A list of the names of the parameters
                that are being inferred. This is only required for
                `multinest`, but we require it for all samplers for
                consistency.
            random_seed: Random seed to use for reproducibility.
            sampler_kwargs: Any additional keyword arguments that should
                be passed to the sampler. Depending on the sampler, this
                might require additional pre-processing, like converting
                a string to a class object.
        """

        # Store the construct arguments
        self.run_dir = run_dir
        self.prior_transform = prior_transform
        self.log_likelihood = log_likelihood
        self.n_dim = n_dim
        self.n_livepoints = n_livepoints
        self.inferred_parameters = inferred_parameters
        self.random_seed = random_seed
        self.sampler_kwargs = sampler_kwargs

        # Initialize attributes
        self.complete = False

        # Save the parameters to a JSON file
        # This is only really required for the MultiNest sampler, but we
        # do it for all samplers for consistency.
        with open(run_dir / "params.json", "w") as json_file:
            json.dump(inferred_parameters, json_file, indent=2)

    @staticmethod
    def _prepare_sampler_kwargs(
        sampler_kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Small utility function to convert the `sampler_kwargs` to a
        dictionary if it is `None`. The `deepcopy()` is required so that
        we can `pop()` the "special" keys without modifying the original
        dictionary.
        """

        return (
            deepcopy(sampler_kwargs) if sampler_kwargs is not None
            else {}
        )

    @abstractmethod
    def run(
        self,
        max_runtime: int,
        verbose: bool = False,
        run_kwargs: dict[str, Any] | None = None,
    ) -> float:
        """
        Run the sampler (for a maximum of `max_runtime` seconds) and
        return the actual runtime of the sampler.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def cleanup(self) -> None:
        """
        Any cleanup that needs to be done after the sampler has
        finished, e.g., closing open pools.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def save_results(self) -> None:
        """
        Save the results of the sampler to the run directory.
        """
        raise NotImplementedError  # pragma: no cover

    def save_runtime(self, runtime: float) -> None:
        """
        Save the runtime of the sampler (in seconds) to a file.
        """
        with open(self.run_dir / "runtime.txt", "a") as f:
            f.write(f"{runtime}\n")

    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def get_weighted_posterior_mean(self) -> np.ndarray:
        """
        Get the weighted posterior mean.
        """

        return np.asarray(
            np.average(self.points, weights=self.weights, axis=0)
        )


class NautilusSampler(Sampler):
    """
    Wrapper around the sampler provided by the `nautilus` package.
    """

    def __init__(
        self,
        run_dir: Path,
        prior_transform: Callable[[np.ndarray], np.ndarray],
        log_likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        inferred_parameters: list[str],
        random_seed: int = 42,
        sampler_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Create a new `NautilusSampler` instance.

        The "special" `sampler_kwargs` (i.e., keyword arguments that
        will be popped and pre-processed instead of being passed to the
        constructor of the sampler directly) are:

            use_pool: A boolean that specifies whether to use a pool
                for parallelization. Default is `True`.
        """

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            sampler_kwargs=sampler_kwargs,
            random_seed=random_seed,
        )

        # Convert the `sampler_kwargs` to a dictionary if it is `None`
        sampler_kwargs = self._prepare_sampler_kwargs(sampler_kwargs)

        # Define the path for the checkpoint file
        self.checkpoint_path = self.run_dir / "checkpoint.hdf5"

        # Import this here to reduce dependencies
        from nautilus import Sampler as _NautilusSampler

        # Set up the pool
        # Setting the `pool` to None will disable parallelization. Otherwise,
        # we need to use the `Pool` from `multiprocess` (instead of the default
        # one from `multiprocessing` that we get if we pass an integer value
        # to the `pool` argument) # because we need the `dill` serializer to
        # send the `log_likelihood` to the worker processes.
        use_pool = sampler_kwargs.pop("use_pool", True)
        self.pool = get_pool() if use_pool else None

        # Note: The argument of `NautilusSampler` is called `likelihood`, but
        # at least according to the docstring, it does indeed expect "the
        # natural logarithm of the likelihood" as its input.
        #
        # noinspection PyTypeChecker
        # noinspection PyUnresolvedReferences
        self.sampler = _NautilusSampler(
            prior=prior_transform,
            likelihood=log_likelihood,  # see above
            n_dim=self.n_dim,
            n_live=self.n_livepoints,
            pool=self.pool,
            filepath=self.checkpoint_path,
            seed=self.random_seed,
            **sampler_kwargs,
        )

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> float:
        """
        Run the Nautilus sampler.
        """

        start_time = time.time()
        run_kwargs = run_kwargs if run_kwargs is not None else {}

        # Run until the timeout is reached...
        self.complete = self.sampler.run(
            verbose=verbose,
            discard_exploration=True,
            timeout=max_runtime,
            **run_kwargs,
        )

        if not self.complete:
            print("\nTimeout reached, stopping sampler!\n")

        # Return the actual runtime
        return time.time() - start_time

    def cleanup(self) -> None:
        for pool in (self.sampler.pool_l, self.sampler.pool_s):
            if pool is not None:
                pool.close()

    def save_results(self) -> None:
        points, log_w, log_l = self.sampler.posterior()
        file_path = self.run_dir / "posterior.npz"
        np.savez(file_path, points=points, log_w=log_w, log_l=log_l)

    @property
    def points(self) -> np.ndarray:
        points, *_ = self.sampler.posterior()
        return np.array(points)

    @property
    def weights(self) -> np.ndarray:
        _, log_w, *_ = self.sampler.posterior()
        return np.array(np.exp(log_w))


class DynestySampler(Sampler):
    """
    Wrapper around the samplers provided by the `dynesty` package.
    """

    def __init__(
        self,
        run_dir: Path,
        prior_transform: Callable[[np.ndarray], np.ndarray],
        log_likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        inferred_parameters: list[str],
        random_seed: int = 42,
        sampler_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Create a new `DynestySampler` instance.

        The "special" `sampler_kwargs` (i.e., keyword arguments that
        will be popped and pre-processed instead of being passed to the
        constructor of the sampler directly) are:

            sampling_mode: This can be either 'standard' to use a
                `dynesty.NestedSampler` or 'dynamic' to use a
                `dynesty.DynamicNestedSampler`. See `dynesty` docs
                for more details.
            use_pool: A dictionary that specifies which parts of the
                nested sampling algorithm should be parallelized. The
                dict should have the following keys: 'propose_point',
                'prior_transform', and 'loglikelihood'. The values
                should be booleans that specify whether the respective
                part of the algorithm should be parallelized.
                If `use_pool` is set to `None`, the default options are
                to use the pool for 'propose_point' and 'loglikelihood',
                but not for 'prior_transform'. (See comments in code.)
        """

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            sampler_kwargs=sampler_kwargs,
            random_seed=random_seed,
        )

        # Convert the `sampler_kwargs` to a dictionary if it is `None`
        sampler_kwargs = self._prepare_sampler_kwargs(sampler_kwargs)

        # Import this here to reduce dependencies
        import dynesty.utils

        # Handle the `sampling_mode` argument
        sampling_mode = sampler_kwargs.pop("sampling_mode", "standard")
        if sampling_mode == "standard":
            from dynesty import NestedSampler as _DynestySampler
        elif sampling_mode == "dynamic":
            from dynesty import DynamicNestedSampler as _DynestySampler
        else:  # pragma: no cover
            raise ValueError(f"{sampling_mode=} is not a valid choice!")

        # Use `dill` instead of `pickle` for serialization; this seems to fix
        # an issue with the `DynestySampler` that pops up when the sampler is
        # resumed from a checkpoint and then tries to create a new checkpoint.
        # Apparently, `dynesty.utils.pickle_module = dill` is not enough; that
        # is why we also construct the `pool` manually.
        dynesty.utils.pickle_module = dill
        self.pool = get_pool()

        # Set up the default pool options:
        # In principle, we could simply pass the `use_pool` dictionary from the
        # `sampler_kwargs` directly to the constructor of the sampler, but this
        # bit of code allows us to overwrite the default settings.
        # Depending on the exact retrieval that one is running, it can make
        # sense to enable to disable some of these options. For example, the
        # toy example seems to run the fastest if only the `loglikelihood`
        # is parallelized; however, for the petitRADTRANS retrievals, we want
        # also want to enable the option for `propose_point`. More experiments
        # might be required to understand what are the best options here.
        use_pool = sampler_kwargs.pop("use_pool", None)
        self.use_pool = (
            use_pool
            if use_pool is not None
            else {
                "propose_point": True,
                "prior_transform": False,
                "loglikelihood": True,
            }
        )

        # Define the path for the checkpoint file
        self.checkpoint_path = self.run_dir / "checkpoint.save"
        self.resume = self.checkpoint_path.exists()

        # Resuming from a checkpoint requires a different initialization than
        # starting a new run...
        if self.resume:
            self.sampler = _DynestySampler.restore(
                fname=self.checkpoint_path.as_posix(),
                pool=self.pool,
            )
        else:
            # noinspection PyTypeChecker
            self.sampler = _DynestySampler(
                loglikelihood=self.log_likelihood,
                prior_transform=self.prior_transform,
                ndim=self.n_dim,
                nlive=self.n_livepoints,
                pool=self.pool,
                use_pool=self.use_pool,
                queue_size=get_number_of_available_cores(),
                rstate=np.random.Generator(np.random.PCG64(self.random_seed)),
                **sampler_kwargs,
            )

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> float:
        """
        Run the dynesty sampler.
        """

        start_time = time.time()
        run_kwargs = run_kwargs if run_kwargs is not None else {}

        # Treat warnings as errors to catch them via try/except
        warnings.filterwarnings("error")

        try:
            with timelimit(max_runtime):
                self.sampler.run_nested(
                    checkpoint_file=self.checkpoint_path.as_posix(),
                    print_progress=verbose,
                    resume=self.resume,
                    **run_kwargs,
                )
        except TimeoutException:
            print("\nTimeout reached, stopping sampler!\n")
            return time.time() - start_time
        except UserWarning as e:
            if "You are resuming a finished static run" in str(e):
                self.complete = True
                return time.time() - start_time
            if "The sampling was stopped short due to maxiter" in str(e):
                self.complete = True
                return time.time() - start_time
            else:  # pragma: no cover
                raise e
        finally:
            warnings.resetwarnings()

        self.complete = True
        return time.time() - start_time

    def cleanup(self) -> None:
        self.pool.close()

    def save_results(self) -> None:
        file_path = self.run_dir / "posterior.pickle"
        with open(file_path, "wb") as handle:
            dill.dump(obj=self.sampler.results, file=handle)

    @property
    def points(self) -> np.ndarray:
        return np.array(self.sampler.results.samples)

    @property
    def weights(self) -> np.ndarray:
        return np.array(self.sampler.results.importance_weights())


class MultiNestSampler(Sampler):
    """
    Wrapper around the sampler provided by the `pymultinest` package.
    """

    def __init__(
        self,
        run_dir: Path,
        prior_transform: Callable[[np.ndarray], np.ndarray],
        log_likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        inferred_parameters: list[str],
        random_seed: int = 42,
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a new `DynestySampler` instance.

        Given that the MultiNest sampler does not construct a `sampler`
        object, any `sampler_kwargs` will effectively be ignored. The
        behavior of the MultiNest sampler is controlled by the arguments
        passed to the `run()` method, i.e., the `run_kwargs`.
        """

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            sampler_kwargs=sampler_kwargs,
            random_seed=random_seed,
        )

        # Handle caching of points and weights that need to be loaded from
        # the MultiNest output files (we only want to read them once)
        self._points: np.ndarray
        self._weights: np.ndarray
        self._points_and_weights_loaded = False

        self.outputfiles_basename = (self.run_dir / "run").as_posix()

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> float:
        """
        Run the MultiNest sampler.
        """

        start_time = time.time()
        run_kwargs = run_kwargs if run_kwargs is not None else {}

        # Import this here to reduce dependencies
        # MultiNest is a pain to install, so we only import it if we need it
        from pymultinest.solve import solve as _solve_pymultinest

        # We cannot use the usual `timelimit` context manager here because
        # MultiNest simply ignores the Exception and continues running. As
        # a workaround, we use a separate process and terminate it if the
        # timeout is reached. We use `pathos.multiprocessing` instead of
        # `multiprocessing` and wrap the function call in a `partial` to
        # make sure that the function can be pickled. This is probably not
        # the most elegant solution, but it works...

        # noinspection PyUnresolvedReferences
        process = multiprocess.Process(
            target=partial(
                _solve_pymultinest,
                LogLikelihood=self.log_likelihood,
                Prior=self.prior_transform,
                n_dims=self.n_dim,
                outputfiles_basename=self.outputfiles_basename,
                n_live_points=self.n_livepoints,
                verbose=verbose,
                resume=True,
                seed=self.random_seed,
                **run_kwargs,
            ),
        )
        process.start()
        process.join(timeout=max_runtime)

        if process.is_alive():
            process.terminate()
            print("Timeout reached, stopping sampler!")
        else:
            self.complete = True

        # Return the actual runtime
        return time.time() - start_time

    def cleanup(self) -> None:
        pass

    def save_results(self) -> None:
        pass  # all results are saved automatically

    def _load_points_and_weights(self) -> None:
        """
        Load the points and weights from the MultiNest output files.
        """

        # Skip if we have already loaded the points and weights
        if self._points_and_weights_loaded:
            return

        # Import this here to reduce dependencies
        from pymultinest.analyse import Analyzer

        # Load the posterior samples from the MultiNest output files
        # We locally redirect stdout to /dev/null to suppress the output
        with contextlib.redirect_stdout(None):
            analyzer = Analyzer(
                n_params=self.n_dim,
                outputfiles_basename=self.outputfiles_basename,
            )
            posterior_samples = analyzer.get_equal_weighted_posterior()[:, :-1]

        self._points = np.array(posterior_samples)
        self._weights = np.ones(len(posterior_samples))
        self._points_and_weights_loaded = True

    @property
    def points(self) -> np.ndarray:
        self._load_points_and_weights()
        return self._points

    @property
    def weights(self) -> np.ndarray:
        self._load_points_and_weights()
        return self._weights


class UltraNestSampler(Sampler):
    """
    Wrapper around the sampler provided by the `ultranest` package.
    """

    def __init__(
        self,
        run_dir: Path,
        prior_transform: Callable[[np.ndarray], np.ndarray],
        log_likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        inferred_parameters: list[str],
        random_seed: int = 42,
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a new `UltraNestSampler` instance.

        Currently, this sampler does not implement any "special"
        `sampler_kwargs` (i.e., keyword arguments that will be popped
        and pre-processed instead of being passed to the constructor
        of the sampler directly).
        """

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            random_seed=random_seed,
        )

        # Convert the `sampler_kwargs` to a dictionary if it is `None`
        sampler_kwargs = self._prepare_sampler_kwargs(sampler_kwargs)

        # Import this here to reduce dependencies
        from ultranest import ReactiveNestedSampler as _UltraNestSampler

        # Create the sampler
        # noinspection PyTypeChecker
        self.sampler = _UltraNestSampler(
            param_names=inferred_parameters,
            loglike=log_likelihood,
            transform=prior_transform,
            log_dir=run_dir.as_posix(),
            resume="resume",
            vectorized=False,
            storage_backend="hdf5",
            **sampler_kwargs,
        )

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> float:
        """
        Run the ultranest sampler.
        """

        # IMPORTANT: Do *NOT* fix the random seed of numpy's global RNG here,
        # at least not without using the `rank` as a seed offset. If you do,
        # this will result in all processes generating the same random numbers
        # (in particular: the same live points), which breaks the sampler!

        # Get the rank of the current process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Convert the `run_kwargs` to a dictionary if it is `None`
        run_kwargs = run_kwargs if run_kwargs is not None else {}

        # Handle the `region_class` argument
        # The default region class is `ultranest.mlfriends.MLFriends`, but we
        # can also pass another choice (like `RobustEllipsoidRegion`) as a
        # string argument in the `run_kwargs()` dictionary, which will be
        # converted to the required class object here.
        region_class_name = run_kwargs.pop("region_class", None)
        if region_class_name is not None:
            region_class = getattr(
                import_module("ultranest.mlfriends"),
                str(region_class_name)
            )
        else:
            from ultranest.mlfriends import MLFriends
            region_class = MLFriends

        # Start the timer
        start_time = time.time()

        # Get the number of likelihood evaluations to run between checking
        # the timeout condition. This is a bit of a "magic number", and the
        # default value is based on the folllowing crude estimate:
        #   96 cores, ~2 sec per likelihood call -> ~48 calls / sec on avg.
        # The total runtime should be within +/- 5 minutes of the max_runtime.
        #   300 sec * 48 calls / sec = 14_400 calls
        # Let's set this number to 10k for now to account for overhead.
        n_calls_between_timeout_checks = run_kwargs.pop(
            "n_calls_between_timeout_checks", 10_000
        )

        # Run the sampler with the given time limit
        while True:

            n_call_before = copy(self.sampler.ncall)

            if rank == 0:
                print("\n\n" + 80 * "-")
                print(f"Calling run() at ncall={n_call_before:,}")
                print(80 * "-" + "\n\n")

            # Run for a given number of likelihood evaluations
            self.sampler.run(
                max_ncalls=n_call_before + n_calls_between_timeout_checks,
                min_num_live_points=self.n_livepoints,
                region_class=region_class,
                **run_kwargs,
            )

            # Check if we have converged
            if self.sampler.ncall == n_call_before:
                self.complete = True
                return time.time() - start_time

            # Check if the timeout is reached
            if time.time() - start_time > max_runtime:
                print("Timeout reached, stopping sampler!")
                return time.time() - start_time

    def cleanup(self) -> None:
        pass

    def save_results(self) -> None:
        file_path = self.run_dir / "posterior.npz"
        np.savez(
            file_path,
            points=self.points,
            weights=self.weights,
        )

    @property
    def points(self) -> np.ndarray:
        return np.array(self.sampler.results["weighted_samples"]["points"])

    @property
    def weights(self) -> np.ndarray:
        return np.array(self.sampler.results["weighted_samples"]["weights"])


# noinspection PyUnresolvedReferences
def get_pool() -> multiprocess.Pool:
    """
    Get a `multiprocess.Pool` with # processes = # available cores.
    """

    # noinspection PyUnresolvedReferences
    return multiprocess.Pool(processes=get_number_of_available_cores())


def get_sampler(name: str) -> Type[Sampler]:
    """
    Convenience function to get a sampler class by name.
    """

    match name:
        case "nautilus":
            return NautilusSampler
        case "dynesty":
            return DynestySampler
        case "multinest":
            return MultiNestSampler
        case "ultranest":
            return UltraNestSampler
        case _:  # pragma: no cover
            raise ValueError(f"Sampler `{name}` not supported!")

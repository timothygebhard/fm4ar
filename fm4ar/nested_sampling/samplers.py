"""
Define abstractions for the different nested sampling implementations.
"""

import contextlib
import json
import time
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Type

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
        **_: Any,  # ignore any additional arguments
    ) -> None:

        self.run_dir = run_dir
        self.prior_transform = prior_transform
        self.log_likelihood = log_likelihood
        self.n_dim = n_dim
        self.n_livepoints = n_livepoints
        self.inferred_parameters = inferred_parameters
        self.random_seed = random_seed

        self.complete = False

        # Save the parameters to a JSON file
        # This is only really required for the MultiNest sampler, but we
        # do it for all samplers for consistency.
        with open(run_dir / "params.json", "w") as json_file:
            json.dump(inferred_parameters, json_file, indent=2)

    @abstractmethod
    def run(
        self,
        max_runtime: int,
        verbose: bool = False,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Run the sampler for the given `max_runtime`.
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

    def save_runtime(self, start_time: float) -> None:
        runtime = time.time() - start_time
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
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            random_seed=random_seed,
        )

        self.checkpoint_path = self.run_dir / "checkpoint.hdf5"

        # Import this here to reduce dependencies
        from nautilus import Sampler as _NautilusSampler

        # [1] The argument of `NautilusSampler` is called `likelihood`, but at
        # least according to the docstring, it does indeed expect "the natural
        # logarithm of the likelihood" as its input.
        # [2] We need the `Pool` from `multiprocess` (instead of the default
        # one from `multiprocessing` that we get if we pass an in to `pool`)
        # because we need the `dill` serializer to send the `log_likelihood`
        # to the worker processes.
        #
        # noinspection PyTypeChecker
        # noinspection PyUnresolvedReferences
        self.sampler = _NautilusSampler(
            prior=prior_transform,
            likelihood=log_likelihood,  # see [1]
            n_dim=self.n_dim,
            n_live=self.n_livepoints,
            pool=multiprocess.Pool(get_number_of_available_cores()),  # see [2]
            filepath=self.checkpoint_path,
            seed=self.random_seed,
        )

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Run the Nautilus sampler.
        """

        start_time = time.time()
        run_kwargs = run_kwargs if run_kwargs is not None else {}

        try:
            with timelimit(max_runtime):
                self.sampler.run(
                    verbose=verbose,
                    discard_exploration=True,
                    **run_kwargs,
                )
        except TimeoutException:
            print("\nTimeout reached, stopping sampler!\n")
            return
        finally:
            self.save_runtime(start_time)

        self.complete = True

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
        sampling_mode: Literal["standard", "dynamic"] = "dynamic",
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
            random_seed=random_seed,
        )

        # Import this here to reduce dependencies
        import dynesty.utils

        if sampling_mode == "standard":
            from dynesty import NestedSampler as _DynestySampler
        elif sampling_mode == "dynamic":
            from dynesty import DynamicNestedSampler as _DynestySampler
        else:
            raise ValueError(
                "`sampling_mode` must be 'standard' or 'dynamic', "
                f"not '{sampling_mode}'!"
            )

        # Use `dill` instead of `pickle` for serialization; this seems to fix
        # an issue with the `DynestySampler` that pops up when the sampler is
        # resumed from a checkpoint and then tries to create a new checkpoint.
        # Apparently, `dynesty.utils.pickle_module = dill` is not enough; that
        # is why we also construct the `pool` manually.
        dynesty.utils.pickle_module = dill
        self.pool_size = get_number_of_available_cores()
        # noinspection PyUnresolvedReferences
        self.pool = multiprocess.Pool(self.pool_size)

        self.checkpoint_path = self.run_dir / "checkpoint.save"
        self.resume = self.checkpoint_path.exists()

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
                queue_size=self.pool_size,
                rstate=np.random.Generator(np.random.PCG64(self.random_seed)),
            )

    def run(
        self,
        max_runtime: int,
        verbose: bool = True,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Run the Dynesty sampler.
        """

        start_time = time.time()
        run_kwargs = run_kwargs if run_kwargs is not None else {}

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
            return
        except RuntimeWarning as e:
            if "resume the run that has ended successfully." in str(e):
                self.complete = True
                return
            else:
                raise e
        except UserWarning as e:
            if "You are resuming a finished static run" in str(e):
                self.complete = True
                return
            else:
                raise e
        finally:
            self.save_runtime(start_time)

        self.complete = True

    def cleanup(self) -> None:
        self.pool.close()

    def save_results(self) -> None:
        file_path = self.run_dir / "posterior.pickle"
        with open(file_path, "wb") as handle:
            dill.dump(obj=self.sampler.results, file=handle)

    @property
    def points(self) -> np.ndarray:
        return np.array(self.sampler.results["samples"])

    @property
    def weights(self) -> np.ndarray:
        return np.array(np.exp(self.sampler.results["logwt"]))


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
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            inferred_parameters=inferred_parameters,
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
    ) -> None:
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
            self.save_runtime(start_time)
            return

        self.complete = True

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
        case _:
            raise ValueError(f"Sampler `{name}` not supported!")

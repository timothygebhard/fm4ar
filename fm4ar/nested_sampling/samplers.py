"""
Define abstractions for the different nested sampling implementations.
"""

import json
import time
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable, Type

import dill
import multiprocess
import numpy as np

from fm4ar.utils.multiproc import get_number_of_available_cores
from fm4ar.utils.timeout import TimeoutException, timelimit


class Sampler(ABC):

    start_time: float
    complete: bool

    def __init__(
        self,
        run_dir: Path,
        prior: Callable[[np.ndarray], np.ndarray],
        likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        parameters: list[str],
        random_seed: int = 42,
    ) -> None:

        self.run_dir = run_dir
        self.prior = prior
        self.likelihood = likelihood
        self.n_dim = n_dim
        self.n_livepoints = n_livepoints
        self.parameters = parameters
        self.random_seed = random_seed

        self.complete = False

        # Save the parameters to a JSON file
        # This is only really required for the MultiNest sampler, but we
        # do it for all samplers for consistency.
        with open(run_dir / 'params.json', 'w') as json_file:
            json.dump(parameters, json_file, indent=2)

    @abstractmethod
    def run(self, max_runtime: int, verbose: bool = False) -> None:
        """
        Run the sampler for the given `max_runtime`.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Any cleanup that needs to be done after the sampler has
        finished, e.g., closing open pools.
        """
        pass

    @abstractmethod
    def save_results(self) -> None:
        """
        Save the results of the sampler to the run directory.
        """
        pass

    def save_runtime(self, start_time: float) -> None:
        runtime = time.time() - start_time
        with open(self.run_dir / "runtime.txt", "a") as f:
            f.write(f"{runtime}\n")

    @property
    @abstractmethod
    def points(self) -> np.ndarray | None:
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray | None:
        pass


class NautilusSampler(Sampler):

    def __init__(
        self,
        run_dir: Path,
        prior: Callable[[np.ndarray], np.ndarray],
        likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        parameters: list[str],
        random_seed: int = 42,
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior=prior,
            likelihood=likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            parameters=parameters,
            random_seed=random_seed,
        )

        self.checkpoint_path = self.run_dir / "checkpoint.hdf5"

        # Import this here to reduce dependencies
        from nautilus import Sampler as _NautilusSampler

        # noinspection PyTypeChecker
        self.sampler = _NautilusSampler(
            prior=prior,
            likelihood=likelihood,
            n_dim=self.n_dim,
            n_live=self.n_livepoints,
            pool=get_number_of_available_cores(),
            filepath=self.checkpoint_path,
            seed=self.random_seed,
        )

    def run(self, max_runtime: int, verbose: bool = True) -> None:
        """
        Run the Nautilus sampler.
        """

        start_time = time.time()

        try:
            with timelimit(seconds=max_runtime):
                self.sampler.run(
                    verbose=verbose,
                    discard_exploration=True,
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

    def __init__(
        self,
        run_dir: Path,
        prior: Callable[[np.ndarray], np.ndarray],
        likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        parameters: list[str],
        random_seed: int = 42,
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior=prior,
            likelihood=likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            parameters=parameters,
            random_seed=random_seed,
        )

        # Import this here to reduce dependencies
        import dynesty.utils
        from dynesty import DynamicNestedSampler as _DynamicNestedSampler

        dynesty.utils.pickle_module = dill

        # We use pathos.multiprocessing instead of multiprocessing because
        # pathos uses dill instead of pickle, which seems to fix a weird issue
        # with the `DynestySampler` that pops up when the sampler is resumed
        # from a checkpoint and then tries to create a new checkpoint.
        # (Apparently, `dynesty.utils.pickle_module = dill` is not enough?)
        import pathos.multiprocessing as multiprocessing

        self.pool_size = get_number_of_available_cores()
        self.pool = multiprocessing.Pool(self.pool_size)

        self.checkpoint_path = self.run_dir / "checkpoint.save"
        self.resume = self.checkpoint_path.exists()

        if self.resume:
            self.sampler = _DynamicNestedSampler.restore(
                fname=self.checkpoint_path.as_posix(),
                pool=self.pool,
            )
        else:
            # noinspection PyTypeChecker
            self.sampler = _DynamicNestedSampler(
                loglikelihood=self.likelihood,
                prior_transform=self.prior,
                ndim=self.n_dim,
                nlive=self.n_livepoints,
                pool=self.pool,
                queue_size=self.pool_size,
            )

    def run(self, max_runtime: int, verbose: bool = True) -> None:
        """
        Run the Dynesty sampler.
        """

        start_time = time.time()

        try:
            with timelimit(seconds=max_runtime):
                self.sampler.run_nested(
                    checkpoint_file=self.checkpoint_path.as_posix(),
                    print_progress=verbose,
                    resume=self.resume,
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
        finally:
            self.save_runtime(start_time)

        self.complete = True

    def cleanup(self) -> None:
        self.pool.close()

    def save_results(self) -> None:
        file_path = self.run_dir / "posterior.pickle"
        with open(file_path, 'wb') as handle:
            dill.dump(obj=self.sampler.results, file=handle)

    @property
    def points(self) -> np.ndarray:
        return np.array(self.sampler.results["samples"])

    @property
    def weights(self) -> np.ndarray:
        return np.array(np.exp(self.sampler.results["logwt"]))


class MultiNestSampler(Sampler):

    def __init__(
        self,
        run_dir: Path,
        prior: Callable[[np.ndarray], np.ndarray],
        likelihood: Callable[[np.ndarray], float],
        n_dim: int,
        n_livepoints: int,
        parameters: list[str],
        random_seed: int = 42,
    ) -> None:

        super().__init__(
            run_dir=run_dir,
            prior=prior,
            likelihood=likelihood,
            n_dim=n_dim,
            n_livepoints=n_livepoints,
            parameters=parameters,
            random_seed=random_seed,
        )

        self.outputfiles_basename = (self.run_dir / "run").as_posix()

    def run(self, max_runtime: int, verbose: bool = True) -> None:
        """
        Run the MultiNest sampler.
        """

        start_time = time.time()

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
                LogLikelihood=self.likelihood,
                Prior=self.prior,
                n_dims=self.n_dim,
                outputfiles_basename=self.outputfiles_basename,
                n_live_points=self.n_livepoints,
                verbose=verbose,
                resume=True,
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

    @property
    def points(self) -> np.ndarray | None:

        # Import this here to reduce dependencies
        from pymultinest.analyse import Analyzer

        # Load the posterior samples from the MultiNest output files
        analyzer = Analyzer(
            n_params=self.n_dim,
            outputfiles_basename=self.outputfiles_basename,
        )
        posterior_samples = analyzer.get_equal_weighted_posterior()[:, :-1]

        return np.array(posterior_samples)

    @property
    def weights(self) -> None:
        return None  # samples are already equally weighted


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

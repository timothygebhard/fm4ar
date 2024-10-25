"""
Load the saved model, draw samples from it, and evaluate the vectorfield
on a grid of positions. Results are saved to CSV files for plotting.
"""

import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.color import lab2rgb
from torchdiffeq import odeint

from fm4ar.models.build_model import build_model


def get_samples(
    model,
    t_max: float,
    context: dict[str, torch.Tensor],
    tolerance: float = 5e-5,
    method: str = "dopri5",
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:

    model.network.eval()

    # Get the number of samples
    num_samples = context["flux"].shape[0]

    # Solve ODE forwards in time to get from theta_0 to theta_1
    with torch.no_grad():
        torch.manual_seed(random_seed)
        theta_0 = model.sample_theta_0(num_samples)
        _, theta_1 = odeint(
            func=partial(model.evaluate_vectorfield, context=context),
            y0=theta_0,
            t=torch.tensor([0, t_max]).to(model.device),
            atol=tolerance,
            rtol=tolerance,
            method=method,
        )

    return theta_0.cpu().numpy(), theta_1.cpu().numpy()


def point_to_color(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Axuliary function to map (x, y) coordinates onto colors.
    """

    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Convert Cartesian to Polar coordinates
    r = np.sqrt(x ** 2 + y ** 2)
    r /= np.max(r)
    theta = np.arctan2(y, x)

    # Map polar to LCH
    L = np.full_like(x, 75)
    C = 100 - r * 100
    H = (theta / (2 * np.pi)) * 360
    H[H < 0] += 360

    # Map LCH to Lab
    H_rad = np.deg2rad(H)
    a = C * np.cos(H_rad)
    b = C * np.sin(H_rad)

    # Convert Lab to sRGB and ensure values are in [0, 1]
    RGB = np.clip(lab2rgb(np.stack((L, a, b), axis=-1)), 0, 1)

    return RGB


if __name__ == "__main__":

    script_start = time.time()
    print("\nDRAW SAMPLES AND EVALUATE VECTORFIELD\n")

    experiment_dir = Path(__file__).parent

    # -------------------------------------------------------------------------
    # Get command line arguments
    # -------------------------------------------------------------------------

    parser = ArgumentParser()
    parser.add_argument(
        "--n-steps",
        type=int,
        default=5,
        help="Number of steps into which to split the time interval [0, 1].",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Number of samples to draw from the model.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load the saved model
    # -------------------------------------------------------------------------

    print("Loading the trained model...", end=" ", flush=True)
    model = build_model(
        experiment_dir=experiment_dir,
        file_path=experiment_dir / "model__stage_0.pt",
        device="auto",
    )
    print("Done!\n", flush=True)

    # -------------------------------------------------------------------------
    # Draw samples and evaluate the vectorfield
    # -------------------------------------------------------------------------

    # Construct output directory for CSV files
    output_dir = experiment_dir / f"output-{args.n_steps}-{args.n_samples}"
    output_dir.mkdir(exist_ok=True)

    # Create a grid of positions at which to evaluate the vectorfield
    X, Y = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
    grid = np.column_stack([X.ravel(), Y.ravel()])

    # Define a dummy context for the model (we effectively trained an
    # unconditional model, but we still need to provide a context).
    # This also defines the number of samples that we draw.
    context_samples = {
        "flux": torch.ones((args.n_samples, 1)).to(model.device),
    }
    context_grid = {
        "flux": torch.ones((len(grid), 1)).to(model.device),
    }

    # Construct values for t
    # Setting t_max=0 breaks the ODE solver, so we use 1e-8 instead
    t_values = np.linspace(0, 1, args.n_steps)
    t_values[0] = 1e-8

    # Loop over different values of t_max
    print("Drawing samples and evaluating vector field:")
    for t in t_values:

        print(f"-- Running for {t=:.2f}...", end=" ", flush=True)

        idx = f"{round(t, 2):.2f}"

        # Draw samples from the model
        base_samples, target_samples = get_samples(
            model=model,
            t_max=t,
            context=context_samples,
        )

        # Keep only samples that are within the bounds of the plot
        mask = np.max(np.abs(base_samples), axis=1) < 2.8

        # Evaluate the vectorfield at the grid positions
        with torch.no_grad():
            vectorfield = model.evaluate_vectorfield(
                t=t,
                # t = t * torch.ones((len(grid), 1)).to(model.device),
                theta_t=torch.from_numpy(grid).float().to(model.device),
                context=context_grid,
            )
            V = vectorfield.cpu().numpy()

        c = point_to_color(x=base_samples[:, 0], y=base_samples[:, 1])

        # Save the vectorfield to a CSV file
        df = pd.DataFrame(
            {
                "x": X.ravel(),
                "y": Y.ravel(),
                "u": V[:, 0],
                "v": V[:, 1],
            },
        )
        df.to_csv(output_dir / f"quiver-{idx}.csv", sep=",", index=False)

        # Save the samples and their colors to a CSV file
        df = pd.DataFrame(
            {
                "x": target_samples[mask, 0],
                "y": target_samples[mask, 1],
                "r": c[mask, 0],
                "g": c[mask, 1],
                "b": c[mask, 2],
            },
        )
        df.to_csv(output_dir / f"samples-{idx}.csv", sep=",", index=False)

        print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

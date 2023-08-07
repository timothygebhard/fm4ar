"""
This script downloads the data for Ardevol-Martinez et al. (2022) from
their GitLab and saves it in the appropriate location.
"""

import time

import requests

from fm4ar.utils.paths import get_datasets_dir

if __name__ == "__main__":
    script_start = time.time()
    print("\nDOWNLOAD DATA FOR ARDEVOL MARTINEZ ET AL. (2022)\n")

    # Create training directory
    print("Creating training directory...", end=" ", flush=True)
    target_dir = get_datasets_dir() / "ardevol-martinez-2022" / "training"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("Done!\n", flush=True)

    # Download training data
    base_url = "https://gitlab.astro.rug.nl/ardevol/exocnn/-/raw/main/Data"
    for suffix in [
        "/metadata.p",
        "/Training_Sets/parameters_type1.npy",
        "/Training_Sets/parameters_type2.npy",
        "/Training_Sets/trans_type1.npy",
        "/Training_Sets/trans_type2.npy",
    ]:
        print(f"Downloading {suffix}...", end=" ", flush=True)

        url = base_url + suffix + "?raw=true"
        response = requests.get(url)
        with open(target_dir / suffix.split("/")[-1], "wb") as f:
            f.write(response.content)

        print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")

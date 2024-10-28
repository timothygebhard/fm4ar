# Flow Matching for Atmospheric Retrievals

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Data availability](https://img.shields.io/badge/Data-Available_on_Edmond-31705e)](https://doi.org/10.17617/3.LYSSVN)

This repository contains the code for the research paper:

> T. D. Gebhard, J. Wildberger, M. Dax, A. Kofler, D. Angerhausen, S. P. Quanz, B. Schölkopf (2024). 
> "Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adaptive Noise Levels." 
> _Accepted for publication at Astronomy & Astrophysics._


---


## 🚀 Quickstart

Installation should generally work by checking out this repository and running `pip install` on it:

```bash
git clone git@github.com:timothygebhard/fm4ar.git ;
cd fm4ar ;
pip install -e .
```

For developer mode (e.g., unit tests, linters, ...), replace the last line with:

```bash
pip install -e ".[dev]"
```


## 🏕 Setting up the environment

The code in here relies on some environmental variables that you need to set:

```bash
export FM4AR_DATASETS_DIR=/path/to/datasets ;
export FM4AR_EXPERIMENTS_DIR=/path/to/experiments ;
```

You might want to add these lines to your `.bashrc` or `.zshrc` file.

Generally, these folders can be subfolders of this repository; however, there may exists scenarios where this is not desirable (e.g., on a cluster).


## 🐭 Tests

This repository comes with a rather extensive set of unit tests (based on [`pytest`](https://pytest.org)). 
After installing `ml4ptp` with the `[develop]` option, the tests can be run as:

```bash
pytest tests
```

You can also use these tests to ensure that the code is still working when you update the dependencies in `pyproject.toml`.


## 📜 Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{Gebhard_2024,
  author   = {Gebhard, Timothy D. and Wildberger, Jonas and Dax, Maximilian and Angerhausen, Daniel and Quanz, Sascha P. and Schölkopf, Bernhard},
  title    = {Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adaptive Noise Levels},
  year     = 2024,
  journal  = {Astronomy \& Astrophysics},
  addendum = {(Accepted)},
}
```


## ⚖️ License and copyright

The code in this repository was written by [Timothy Gebhard](https://github.com/timothygebhard), with contributions from [Jonas Wildberger](https://github.com/jonaswildberger) and [Maximilian Dax](https://github.com/max-dax), and is owned by the [Max Planck Society](https://www.mpg.de/en).
We release it under a BSD-3 Clause License; see LICENSE for more details.

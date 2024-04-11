# Flow matching for atmospheric retrievals


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

The code in here relies on some environmental variables that you need to set:

```bash
export FM4AR_DATASETS_DIR=/path/to/datasets ;
export FM4AR_EXPERIMENTS_DIR=/path/to/experiments ;
```

You might want to add these lines to your `.bashrc` or `.zshrc` file.

Generally, these folders can be subfolders of this repository; however, there may exists scenarios where this is not desirable (e.g., on a cluster).


## 💻 Code style guide

A few comments on the code style (and the tool chain used to enforce it) in this repository:

- Use [**ruff**](https://github.com/charliermarsh/ruff) as a fast Python linter and formatter / import sorter:
    ```bash
    ruff check .
    ruff format .  # double check the changes before committing
    ```
- Use [**mypy**](https://github.com/python/mypy) to check types:
    ```bash
    mypy .
    ```

All tools are configured through the `pyproject.toml` file.


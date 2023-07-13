import torch
import numpy as np
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float32)


class ArDataset(Dataset):
    def __init__(self, theta, x):
        super(ArDataset, self).__init__()
        # standardization
        # only standardizing theta for now, since x may be constant for early times -> nan
        self.standardization = {
            "x": {"mean": torch.zeros_like(x[0]), "std": torch.ones_like(x[0])},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


# TODO: Replace this with our dataset generation
def generate_dataset(task, settings, dataset_size):

    prior = task.get_prior()
    simulator = task.get_simulator()

    theta = np.array(prior.sample((dataset_size,)))
    x = simulator(theta)
    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)

    settings["data"]["dim_theta"] = theta.shape[1]
    settings["data"]["dim_x"] = x.shape[1]
    if "symmetry_kwargs" in settings["model"]["posterior_kwargs"]:
        settings["model"]["posterior_kwargs"]["symmetry_kwargs"]["bins"] = x.shape[1]
        settings["model"]["posterior_kwargs"]["symmetry_kwargs"]["T_total"] = (
            task.time["upper"] - task.time["lower"]
        )

    return ArDataset(theta, x)
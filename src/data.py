from os import path

import torch
import torchvision.transforms as T

from pytorch_lightning import LightningDataModule
from pytorch_lightning.demos.mnist_datamodule import MNIST

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        MNIST(DATASETS_PATH, download=True)

    def train_dataloader(self):
        train_dataset = MNIST(DATASETS_PATH, train=True, download=False, transform=self.transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        test_dataset = MNIST(DATASETS_PATH, train=False, download=False, transform=self.transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size)
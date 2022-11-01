import torch
from torch.nn import functional as F
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule

from pytorch_lightning.demos.boring_classes import Net


class ImageClassifier(LightningModule):
    def __init__(self, lr=1.0, gamma=0.7, batch_size=32, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]



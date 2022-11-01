from src.data import MNISTDataModule
from src.model import ImageClassifier
from dvclive.lightning import DvcLiveLogger
import argparse
from pytorch_lightning import Trainer


def main(args):
    logger = DvcLiveLogger(run_name='test-0', path='dvc_logs')
    system = ImageClassifier(**vars(args))
    dm = MNISTDataModule(batch_size=args.batch_size)
    trainer = Trainer(logger=logger, gpus=2, strategy='ddp', accelerator='gpu', max_epochs=10)
    trainer.fit(model=system, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    hparams = parser.parse_args()
    main(hparams)

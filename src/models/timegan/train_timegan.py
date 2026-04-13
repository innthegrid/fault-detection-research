# src/models/timegan/train_timegan.py
import os
import torch
import numpy as np
import argparse
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.timegan.model import TimeGAN
from src.data_utils.loaders import get_dataloaders


# ==========================================================
# Reproducibility
# ==========================================================
SEED = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# MAIN
# ==========================================================
def main(hparams):

    # ----------------------------
    # REQUIRED: model type flag
    # ----------------------------
    hparams.model_type = "timegan"

    # ----------------------------
    # Experiment setup (FAULT-AGNOSTIC)
    # ----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_tag = hparams.train_dir.strip("/").replace("/", "_")
    test_tag = hparams.test_dir.strip("/").replace("/", "_")

    run_name = f"timegan_{train_tag}_vs_{test_tag}_{timestamp}"

    exp_dir = os.path.join("./experiments", run_name)

    for sub in ["ckpt", "logs", "result"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    hparams.exp_dir = exp_dir

    print(f"Starting TimeGAN Experiment: {run_name}")

    # ----------------------------
    # Logger + Model
    # ----------------------------
    logger = TensorBoardLogger(save_dir=exp_dir, name="logs")
    model = TimeGAN(hparams)

    # ----------------------------
    # Checkpoint (GAN-friendly)
    # ----------------------------
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "ckpt"),
        filename="timegan-{epoch:02d}",
        save_top_k=1,
        monitor=None,
    )

    # ----------------------------
    # Data (FULL MULTI-FAULT SUPPORT)
    # ----------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=hparams.data_dir,
        train_dir=hparams.train_dir,
        test_dir=hparams.test_dir,
        hparams=hparams,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[checkpoint],
        logger=logger,
        accelerator="auto",
        devices=1 if hparams.gpu >= 0 else "auto",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # ----------------------------
    # TRAIN
    # ----------------------------
    print("--- Training Start ---")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\nResults saved in: {exp_dir}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")


# ==========================================================
# CLI ENTRY
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TimeGAN Training Script")

    # ----------------------------
    # REQUIRED DATA INPUTS
    # ----------------------------
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    # ----------------------------
    # MODEL PARAMS
    # ----------------------------
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--module", type=str, default="gru")

    # ----------------------------
    # TRAINING PARAMS
    # ----------------------------
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)

    # ----------------------------
    # PRETRAIN
    # ----------------------------
    parser.add_argument("--pretrain_epochs", type=int, default=10)

    # ----------------------------
    # TEST MODE
    # ----------------------------
    parser.add_argument("--only_test", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()

    # ==========================================================
    # TEST-ONLY MODE
    # ==========================================================
    if args.only_test == 1:

        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")

        args.model_type = "timegan"

        print(f"Loading checkpoint: {args.ckpt_path}")

        model = TimeGAN.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            hparams=args,
        )

        _, _, test_loader = get_dataloaders(
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            hparams=args,
        )

        trainer = Trainer(accelerator="auto", devices=1)

        trainer.test(model, dataloaders=test_loader)

    else:
        main(args)

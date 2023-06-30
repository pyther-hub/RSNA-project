import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import gc
from config import config as Config
from datasets import BreastCancerDataSet, get_transforms
from classifier import BreastCancerModel
from utils import *
from torch.utils.data import WeightedRandomSampler


RSNA_PATH = "data"
TARGET = "cancer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2


def save_model(name, model, thres, model_type):
    torch.save(
        {"model": model.state_dict(), "threshold": thres, "model_type": model_type},
        f"{name}",
    )


def train_model(ds_train, ds_eval, name, config=Config, do_save_model=True):
    torch.manual_seed(42)
    sampler = WeightedRandomSampler(
        weights=ds_train.each_row_weights, num_samples=len(ds_train), replacement=True
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        sampler=sampler,
        pin_memory=True,
    )

    model = BreastCancerModel(config.MODEL_TYPE, config.DROPOUT).to(DEVICE)

    optim = torch.optim.Adam(model.parameters())

    scheduler = None
    if config.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=config.ONE_CYCLE_MAX_LR,
            epochs=config.EPOCHS,
            steps_per_epoch=len(dl_train),
            pct_start=config.ONE_CYCLE_PCT_START,
        )

    scaler = GradScaler()
    best_eval_score = 0
    for epoch in tqdm(range(config.EPOCHS), desc="Epoch"):
        model.train()
        with tqdm(dl_train, desc="Train", mininterval=30) as train_progress:
            for batch_idx, (X, y_cancer) in enumerate(train_progress):
                optim.zero_grad()
                with autocast():
                    y_cancer_pred = model.forward(X.to(DEVICE))
                    cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        y_cancer_pred,
                        y_cancer.to(float).to(DEVICE),
                        pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(
                            DEVICE
                        ),
                    )
                    loss = cancer_loss
                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        print(f"Bad loss, skipping the batch {batch_idx}")
                        del loss, cancer_loss, y_cancer_pred
                        gc_collect()
                        continue

                # scaler is needed to prevent "gradient underflow"
                scaler.scale(loss).backward()
                scaler.step(optim)
                if scheduler is not None:
                    scheduler.step()

                scaler.update()

                lr = (
                    scheduler.get_last_lr()[0] if scheduler else config.ONE_CYCLE_MAX_LR
                )
                print(
                    {
                        "loss": (loss.item()),
                        "cancer_loss": cancer_loss.item(),
                        "lr": lr,
                        "epoch": epoch,
                    }
                )

            (f1, thres), val_preds, loss = evaluate_model(
                model, ds_eval, shuffle=False, config=config
            )

            if f1 > best_eval_score:
                best_eval_score = f1
                if do_save_model:
                    save_model(name, model, thres, config.MODEL_TYPE)

            print(
                {
                    "eval_cancer_loss": cancer_loss,
                    "eval_f1": f1,
                    "max_eval_f1": best_eval_score,
                    "eval_f1_thres": thres,
                    "eval_loss": loss,
                    "epoch": epoch,
                }
            )

    return model


def evaluate_model(model: BreastCancerModel, ds, shuffle=False, config=Config):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    pred_cancer = []
    with torch.no_grad():
        model.eval()
        losses = []
        targets = []
        with tqdm(dl_test, desc="Eval", mininterval=30) as progress:
            for i, (X, y_cancer) in enumerate(progress):
                with autocast(enabled=True):
                    X = X.to(DEVICE)
                    y_cancer_pred = model.forward(X)
                    if config.TTA:
                        y_cancer_pred2 = model.forward(torch.flip(X, dims=[-1]))
                        y_cancer_pred = (y_cancer_pred + y_cancer_pred2) / 2

                    cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        y_cancer_pred,
                        y_cancer.to(float).to(DEVICE),
                        pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(
                            DEVICE
                        ),
                    ).item()

                    pred_cancer.append(torch.sigmoid(y_cancer_pred))
                    losses.append(cancer_loss)
                    targets.append(y_cancer.cpu().numpy())

        targets = np.concatenate(targets)
        pred = torch.concat(pred_cancer).cpu().numpy()
        pf1, thres = optimal_f1(targets, pred)
        return (pf1, thres), pred, np.mean(losses)

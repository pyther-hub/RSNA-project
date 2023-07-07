import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import gc
from utils import *
from datasets import BreastCancerDataSet, get_transforms
from config import Config
from train import train_model

RSNA_PATH = "data"


if __name__ == "__main__":
    df_train = pd.read_csv(f"{RSNA_PATH}/train.csv")
    df_train.age.fillna(df_train.age.mean(), inplace=True)
    df_train["age"] = pd.qcut(df_train.age, 10, labels=range(10), retbins=False).astype(
        int
    )
    TRAIN_IMAGES_PATH = "data/images"
    N_FOLDS = 4
    FOLDS = [x for x in range(0, N_FOLDS)]
    split = StratifiedGroupKFold(N_FOLDS)
    for k, (_, test_idx) in enumerate(
        split.split(df_train, df_train.cancer, groups=df_train.patient_id)
    ):
        df_train.loc[test_idx, "split"] = k
    df_train.split = df_train.split.astype(int)
    df_train.groupby("split").cancer.max()

    for fold in FOLDS:
        gc_collect()
        ds_train = BreastCancerDataSet(
            df_train.query(f"split != fold"),
            TRAIN_IMAGES_PATH,
            get_transforms(aug=Config.AUG),
        )
        ds_eval = BreastCancerDataSet(
            df_train.query(f"split != fold"),
            TRAIN_IMAGES_PATH,
            get_transforms(aug=False),
        )
        train_model(ds_train, ds_eval, f"model-f{fold}")

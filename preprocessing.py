from concurrent.futures import ProcessPoolExecutor
import cv2
import glob
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import re


def fit_image(fname):
    X = cv2.imread(fname)
    X = X[5:-5, 5:-5]
    output = cv2.connectedComponentsWithStats(
        (X > 20).astype(np.uint8)[:, :, 0], 8, cv2.CV_32S
    )
    stats = output[2]
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h
    X_fit = X[y1:y2, x1:x2]

    patient_id, im_id = re.findall("(\d+)_(\d+).png", os.path.basename(fname))[0]
    os.makedirs(patient_id, exist_ok=True)
    cv2.imwrite(f"{patient_id}/{im_id}.png", X_fit[:, :, 0])


def fit_all_images(all_images):
    with ProcessPoolExecutor(2) as p:
        for i in tqdm(p.map(fit_image, all_images), total=len(all_images)):
            pass


all_images = glob.glob("/kaggle/input/rsna-breast-cancer-pngs/*")
DEBUG = True

if DEBUG:
    all_images = np.random.choice(all_images, size=100)

np.random.seed(123)

for fname in np.random.choice(glob.glob("*/*"), size=100):
    plt.figure(figsize=(20, 10))
    patient_id, im_id = re.findall("(\d+)/(\d+).png", fname)[0]
    plt.suptitle(f"[{fname}]")
    im1 = Image.open(fname).convert("F")
    plt.subplot(121).imshow(im1)
    plt.subplot(121).set_title(f"Output image {im1.size}")
    im2 = Image.open(
        f"/kaggle/input/rsna-breast-cancer-pngs/{patient_id}_{im_id}.png"
    ).convert("F")
    plt.subplot(122).imshow(im2)
    plt.subplot(122).set_title(f"Source image {im2.size}")

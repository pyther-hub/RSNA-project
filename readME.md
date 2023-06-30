
# RSNA Screening Mammography Breast Cancer Detection

## What All is Done 

* Model

  * Used simple 2 model used and made an Ensemble
  * tf_efficientnetv2_s
  * maxvit_tiny_tf_384.in1k
* CV strategy

  * used 5 stratifiedgroupkfold using (Patient id and label )
  * ROI extraction was performed using rule-based method
  * The resolution was set to 1024x512 for efficientnet-b3 and b4.
  * The channel number was set to 3 (to use pretrained models)
  * Min-max scaling (-1.0 ~ 1.0)
* Dealing with Imbalance Data

  * Batch size of 8. Adjusted to have a majority (Not Cancer) to minority (Cancer) ratio of 7:1 for each batch. This is essentially oversampling.
* Augmentations, I preferred simple augmentation technique.One god thing was border_mode of ShiftScaleRotate in Albumentations is cv2.BORDER_REFLECT_101

  ```python
  import albumentations as A

  A.HorizontalFlip(p=0.5)
  A.VerticalFlip(p=0.5)

  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8)
  A.OneOf([
      A.RandomGamma(gamma_limit=(50, 150), p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5)
  ], p=0.5)
  A.CoarseDropout(max_height=8, max_width=8, p=0.5)
  ```

### Model Parameters

1. Dropout rate (0.7), A high grouput rate performed better
2. Loss function used was BCEWithLogitsLoss, FocalLoss was not working
3. optimizer: Adam (lr: 1.0e-4)
4. scheduler: OneCycleLR (pct_start: 0.1, div_factor: 1.0e+3, max_lr: 1.0e-4)
5. epoch: 5
6. batch_size: 8 (and accumulate_grad_batches=4, so 8*4=32)
7. fp16 (training and inference)

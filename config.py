class Config:
    ONE_CYCLE = True
    ONE_CYCLE_PCT_START = 0.1
    ADAMW_DECAY = 0.024
    ONE_CYCLE_MAX_LR = 0.0004
    EPOCHS = 5
    MODEL_TYPE = "efficientnet_b3"
    DROPOUT = 0.8
    AUG = True
    POSITIVE_TARGET_WEIGHT = 20
    BATCH_SIZE = 8
    ACCUMULATION_STEPS=4
    AUTO_AUG_M = 10
    AUTO_AUG_N = 2
    TTA = False

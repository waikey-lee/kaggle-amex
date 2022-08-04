# ============================================================================
# Define Path
# ============================================================================
import os

RAW_DATA_PATH = "../raw_data"
INTERIM_DATA_PATH = "../interim_data"
PROCESSED_DATA_PATH = "../processed_data"
SUBMISSION_DATA_PATH = "../submissions"
EVALUATION_DATA_PATH = "../evaluation_data"
MODELS_PATH = "../models"
EXP_PATH = "../experiments"

RAW_TRAIN_PARQUET_PATH = os.path.join(RAW_DATA_PATH, "train_parquet")
RAW_TRAIN_PICKLE_PATH = os.path.join(RAW_DATA_PATH, "train_pickle")
RAW_TEST_PARQUET_PATH = os.path.join(RAW_DATA_PATH, "test_parquet")
RAW_TEST_PICKLE_PATH = os.path.join(RAW_DATA_PATH, "test_pickle")

INTERIM_TRAIN_PARQUET_PATH = os.path.join(INTERIM_DATA_PATH, "train_parquet")
INTERIM_TEST_PARQUET_PATH = os.path.join(INTERIM_DATA_PATH, "test_parquet")
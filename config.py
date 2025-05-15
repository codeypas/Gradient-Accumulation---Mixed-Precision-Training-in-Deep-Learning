import torch

TRAINING_FILE = "data/train.csv"
MODEL_PATH = "model.bin"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
BERT_PATH = "bert-base-uncased"
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection, metrics
from transformers import AdamW, get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import BERTBaseUncased

def run():
    print("Current working directory:", os.getcwd())
    
    # Load dataset
    if not os.path.exists(config.TRAINING_FILE):
        raise FileNotFoundError(f"Training file not found: {config.TRAINING_FILE}")
    
    # dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")

    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    print("Columns in CSV:", dfx.columns)


    # Encode target
    # dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    dfx["sentiment"] = dfx["sentiment"].apply(lambda x: 1 if x.lower() == "positive" else 0)


    # Split dataset
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # Dataset and DataLoader
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )
    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # Model setup
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    # Optimizer and Scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Training Loop
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy:.4f}")

        if accuracy > best_accuracy:
            print("Saving best model...")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    run()

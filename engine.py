import torch
import torch.nn as nn
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        targets = d["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    final_outputs = []
    final_targets = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            final_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())
            final_targets.extend(targets.cpu().numpy().tolist())
    return final_outputs, final_targets

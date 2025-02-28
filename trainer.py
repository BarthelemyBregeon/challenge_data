import torch
from tqdm import tqdm

def train_epoch(model, optimizer, loss_fn, dataloader, device):
    model.train()
    running_loss = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        optimizer.zero_grad()
    return running_loss / len(dataloader)

def test(model, loss_fn, test_dataloader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            running_loss += loss.item()
    return running_loss / len(test_dataloader)

def train(model, optimizer, loss_fn, train_dataloader, device, epochs=10):
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, loss_fn, train_dataloader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")


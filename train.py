import os
import torch
from functions import *
from tqdm import tqdm
from prettytable import PrettyTable

def train_model(model, train_batches, val_batches, optimizer, criterion, device, base_dir, num_epochs=10):
    save_path = make_timestamped_dir(base_dir)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
    best_val_loss = float('inf')
    table = PrettyTable(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_batches)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_batches)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)
        table.add_row([epoch+1, f"{train_loss:.4f}", f"{train_acc:.2f}%", f"{val_loss:.4f}", f"{val_acc:.2f}%"])
        print(table)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'last_epoch_model.pth'))
    return model, save_path
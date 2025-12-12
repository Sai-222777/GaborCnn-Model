# main.py
import time
import torch
import torch.nn as nn
from functions import *
from Model import *
from train import train_model
from test import test_model

torch.manual_seed(1729)

def main():
    device = get_device()
    print(device)
    batch_size = 16
    num_epochs = 60
    learning_rate = 1e-3
    base_model_dir = "./saved_models"

    config = ModelConfig(img_height=512, img_width=512, channels=1, num_classes=4, batch_size=batch_size, epochs=num_epochs)
    use_trained_model = False
    perform_testing = True
    load_model_path = "saved_models/16/best_model.pth"

    train_batches, val_batches, class_indices = create_full_dataset(
    train_dir="dataset/Training",
    val_dir="dataset/Testing",
    config=config
    )

    model = EfficientNet(output=1280,num_filters=16,quantize=False,cluster=False).to(device)
    model_summary(model)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_trained_model:
        best_model = load_trained_model(model, model_path=load_model_path, device=device)

    else:
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared before Training.")
        start_time = time.time()
        best_model, _ = train_model(model, train_batches, val_batches, optimizer, criterion, device, base_model_dir, num_epochs)
        end_time = time.time()
        print(f"⏱️ Training time: {end_time - start_time:.4f} seconds")

    if perform_testing:
        # del train_batches, val_batches
        # test_batches = load_batches(batch_size=batch_size, device=device, test_dataset=True)
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared before testing.")
        test_model(best_model, val_batches, criterion, device)

        

if __name__ == "__main__":
    main()

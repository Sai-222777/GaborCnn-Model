import os
import time
import torch
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict


import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.cluster import KMeans

class ModelConfig:
    """Configuration class for model hyperparameters"""
    def __init__(self, **kwargs):
        self.img_height = kwargs.get("img_height", 512)
        self.img_width = kwargs.get("img_width", 512)
        self.channels = kwargs.get("channels", 1)  # Grayscale or RGB
        self.num_classes = kwargs.get("num_classes", 4)
        self.batch_size = kwargs.get("batch_size", 16)
        self.epochs = kwargs.get("epochs", 50)

class CustomImageDataset(Dataset):
    def __init__(self, df, config: ModelConfig, augment=False):
        self.filepaths = df.filepath.values
        self.labels = df.label.values
        self.config = config
        self.augment = augment

        def zscore_normalize(img_tensor):
            mean = img_tensor.mean()
            std = img_tensor.std()
            return (img_tensor - mean) / (std + 1e-7)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomAffine(
                    degrees=0,
                    shear=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomResizedCrop((config.img_height, config.img_width), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),  # Add Gaussian noise
                transforms.Lambda(zscore_normalize),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.img_height, config.img_width)),
                transforms.ToTensor(),
                transforms.Lambda(zscore_normalize),
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        if self.config.channels == 1:
            image = image.convert("L")  # Grayscale
        else:
            image = image.convert("RGB")  # RGB

        image = self.transform(image)
        return image, label

def load_image_paths_and_labels(base_dir):
    """Mimics TensorFlow _load_image_paths_and_labels"""
    data = []
    class_names = sorted(os.listdir(base_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(class_path, fname)
                data.append((full_path, class_name))

    df = pd.DataFrame(data, columns=["filepath", "class_name"])
    df["label"] = df.class_name.map(label_map)
    return df, label_map

def create_full_dataset(train_dir: str, val_dir: str, config: ModelConfig):
    train_df, class_indices = load_image_paths_and_labels(train_dir)
    val_df, _ = load_image_paths_and_labels(val_dir)

    train_dataset = CustomImageDataset(train_df, config=config, augment=True)
    val_dataset = CustomImageDataset(val_df, config=config, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, class_indices

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_summary(model, shape=(1, 512, 512)):
    summary(model, input_size=shape)

def to_tensor(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return images, labels

def fetch_dataset(val_split=0.00, save_dir='./preprocessed_data'):
    os.makedirs(save_dir, exist_ok=True)

    # Skip processing if all files exist
    train_pt = os.path.join(save_dir, 'train.pt')
    val_pt = os.path.join(save_dir, 'val.pt')
    test_pt = os.path.join(save_dir, 'test.pt')

    if os.path.exists(train_pt) and os.path.exists(val_pt) and os.path.exists(test_pt):
        print(f"✅ Preprocessed data already exists in {save_dir}")
        return
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_train = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform)

    n = len(full_train)
    n_val = int(n * val_split)
    n_train = n - n_val
    train_data, val_data = random_split(full_train, [n_train, n_val])

    torch.save(to_tensor(train_data), os.path.join(save_dir, 'train.pt'))
    torch.save(to_tensor(val_data), os.path.join(save_dir, 'val.pt'))
    torch.save(to_tensor(test), os.path.join(save_dir, 'test.pt'))

    print(f"✅ Data saved in {save_dir}")

def split_into_batches(images, labels, batch_size):
    return [(images[i:i+batch_size], labels[i:i+batch_size]) for i in range(0, len(images), batch_size)]

def load_batches(batch_size=64, device='cuda', save_dir='./preprocessed_data', test_dataset = False):
    if(test_dataset is False):
        train_x, train_y = torch.load(os.path.join(save_dir, 'train.pt'), map_location=device)
        val_x, val_y = torch.load(os.path.join(save_dir, 'val.pt'))

        return (
            split_into_batches(train_x, train_y, batch_size),
            split_into_batches(val_x, val_y, batch_size),
        )
    else:
        test_x, test_y = torch.load(os.path.join(save_dir, 'test.pt'), map_location=device)
        return split_into_batches(test_x, test_y, batch_size)

def make_timestamped_dir(base_path='./saved_models'):
    timestamp = time.strftime("%d-%b-%Y-%I-%M-%S-%p")
    full_path = os.path.join(base_path, timestamp)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def load_trained_model(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded model weights from: {model_path}")
    return model

def quantize_gabor_number(value,fractional_width):

    if(abs(value) == 1):
        return value

    b = ''
    
    if(value < 0):
        b = b + '10'
        value = -value
    else:
        b = b + '00'

    for i in range(1,fractional_width+1):
        value = value * 2
        if(value >= 1):
            b = b + '1'
            value = value - 1
        else:
            b = b + '0'
    
    fractional_value = 0.0
    for i in range(1, fractional_width+1):
        if(b[i+1] == '1'):
            fractional_value += 2 ** (-i)

    if b[0] == '1':
        fractional_value = -fractional_value
    
    return fractional_value

def get_modified_kernel(original_kernel,n_clusters,ksize):
    kernel_flat = original_kernel.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    kmeans.fit(kernel_flat)
    centroids = kmeans.cluster_centers_
    new_kernel_flat = centroids[kmeans.labels_]
    return new_kernel_flat.reshape(ksize)

def get_modified_kernel_with_precision(original_kernel, n_clusters, ksize, bit_widths):
    kernel_flat = original_kernel.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    kmeans.fit(kernel_flat)
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        centroids[i] = quantize_gabor_number(centroids[i],bit_widths[i])
    new_kernel_flat = centroids[kmeans.labels_]
    return new_kernel_flat.reshape(ksize)

def make_balanced_subset(dataset, samples_per_class):
    class_to_indices = defaultdict(list)

    # collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    # build balanced index list
    balanced_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(f"Class {label} has fewer than {samples_per_class} samples.")
        balanced_indices.extend(indices[:samples_per_class])

    return torch.utils.data.Subset(dataset, balanced_indices)

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from collections import Counter

class MaskDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, mask_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        mask = Image.open(row['mask_path']).convert('L')
        label = torch.tensor(row['class'], dtype=torch.float32)
        sphericity = torch.tensor([row['sphericity']], dtype=torch.float32)  # <-- met entre crochets pour avoir [1]
        if self.image_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
        return image, mask, label, sphericity

    def __len__(self):
        return len(self.df)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx]['label']
        return image, label

    def __len__(self):
        return len(self.df) 

def get_dataloaders(batch_size=16, shuffle=True):
    image_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    df = pd.read_csv("/home/dsplab/Robin/image_mask/dataframe/model/dataset_sphericity.csv")    
    df = df.dropna(subset=['sphericity'])
    df['sphericity'] = df['sphericity'].astype(float)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

    train_set = MaskDataset(train_df, image_transform=image_transform, mask_transform=mask_transform)
    val_set = MaskDataset(val_df, image_transform=image_transform, mask_transform=mask_transform)
    test_set = MaskDataset(test_df, image_transform=image_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def show_accuracy_img_sph(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks, labels, sphericity in dataloader:
            images = images.to(device)
            sphericity = sphericity.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)
            outputs = model(images, sphericity)  # Correction: pass sphericity
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            
            total += labels.size(0)
    return correct / total if total > 0 else 0

def show_accuracy_img(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks, labels, sphericity in dataloader:
            images = images.to(device)
            sphericity = sphericity.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)

            outputs = model(images)  # Only_pic
            preds = outputs.argmax(dim=1)  # preds: [batch_size]
            correct += (preds == labels).sum().item()
            
            total += labels.size(0)
    return correct / total if total > 0 else 0

def count_errors_by_class(model, dataloader, device=None, class_names=None):
    """
    Print the number of prediction errors per class.
    Args:
        model: PyTorch model
        dataloader: DataLoader to evaluate (batch: image, mask, label, sphericity)
        device: cpu or cuda
        class_names: list of class names (optional)
    """
    import torch
    from collections import Counter

    if device is None:
        device = torch.device("cpu")
    model.eval()
    error_counter = Counter()
    total_counter = Counter()

    with torch.no_grad():
        for images, masks, labels, sphericity in dataloader:
            images = images.to(device)
            sphericity = sphericity.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)
            outputs = model(images, sphericity)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                total_counter[int(true)] += 1
                if int(true) != int(pred):
                    error_counter[int(true)] += 1

    print("Number of errors per class:")
    for cls in sorted(total_counter.keys()):
        cls_name = class_names[cls] if class_names else str(cls)
        n_errors = error_counter[cls]
        n_total = total_counter[cls]
        print(f"Class {cls_name}: {n_errors} errors out of {n_total} samples ({n_errors/n_total:.2%})")
    return error_counter

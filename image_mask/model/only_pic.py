import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from dataloader import ImageDataset, get_dataloaders, show_accuracy_img
from image_mask.model.LF_models import PyTorchCNN, Custominceptiont
from sklearn.model_selection import train_test_split

# Charge le DataFrame unique
df = pd.read_csv("/home/dsplab/Robin/image_mask/dataframe/model/dataset_sphericity.csv")

# Split train/val/test (exemple 70/15/15)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

# Optionnel: ajoute tes transforms ici
transform = None

train_dataset = ImageDataset(train_df, transform=transform)
val_dataset = ImageDataset(val_df, transform=transform)
test_dataset = ImageDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = batch[:2]
            images = images.to(device)
            labels = labels.long().to(device)
            if labels.ndim > 1:
                labels = labels.view(labels.size(0))

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = batch[:2] 
                images = images.to(device)
                labels = labels.long().to(device)
                if labels.ndim > 1:
                    labels = labels.view(labels.size(0))

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            print("Saving new best model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best val loss: {:.4f}".format(best_val_loss))


if __name__ == "__main__":

    print("Torch CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = Custominceptiont()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=10,
    )

    train_acc = show_accuracy_img(model, train_loader, device=device)
    val_acc = show_accuracy_img(model, val_loader, device=device)
    test_acc = show_accuracy_img(model, test_loader, device=device)
    print(
        f"Train Acc {train_acc*100:.6f}%"
        f" | Val Acc {val_acc*100:.6f}%"
        f" | Test Acc {test_acc*100:.6f}%"
    )

PATH = "plain-pytorch.pt"
torch.save(model.state_dict(), PATH)
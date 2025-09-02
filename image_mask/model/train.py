import torch
import torch.nn as nn
from tqdm import tqdm
from image_mask.model.LF_models import LateFusionModel

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks, labels, sphericity in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            sphericity = sphericity.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)

            outputs = model(images, sphericity)
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
            for images, masks, labels, sphericity in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                images = images.to(device)
                sphericity = sphericity.float().unsqueeze(1).to(device)
                labels = labels.float().to(device)

                outputs = model(images, sphericity)
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


# Exemple d'utilisation depuis ton main.py ou notebook :
if __name__ == "__main__":
    import torchvision.models as models
    from dataloader import MaskDataset, get_dataloaders, show_accuracy_img_sph, count_errors_by_class
    import pandas as pd

    print("Torch CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LateFusionModel(num_tabular_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, shuffle=True)

    train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          criterion, 
          device, 
          num_epochs=10, 
          save_path="best_latefusion_model.pth")
    
    train_acc = show_accuracy_img_sph(model, train_loader, device=device)
    val_acc = show_accuracy_img_sph(model, val_loader, device=device)
    test_acc = show_accuracy_img_sph(model, test_loader, device=device)
    print(
        f"Train Acc {train_acc*100:.6f}%"
        f" | Val Acc {val_acc*100:.6f}%"
        f" | Test Acc {test_acc*100:.6f}%"
    )
    count_errors_by_class(model, test_loader, device=device, class_names=["benign", "malign"])
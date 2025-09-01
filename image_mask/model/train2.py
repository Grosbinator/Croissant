import torch
import torch.nn as nn
from tqdm import tqdm
from cnn import LateFusionModel
import torchvision.models as models
import pytorch_lightning as L
from dataloader import MaskDataset, get_dataloaders, show_accuracy_img_sph, count_errors_by_class
from cnn import LateFusionLightning  
import pandas as pd

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

if __name__ == "__main__":

    

    print("Torch CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prépare tes options pour le modèle et l'entraînement
    class DummyOpts:
        eval_threshold = 0.5
        class loss_opts:
            name = "FocalLoss"
        lr = 1e-4

    model_opts = DummyOpts()
    train_par = DummyOpts()

    model = LateFusionLightning(model_opts, train_par).to(device)  

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, shuffle=True)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    errors = count_errors_by_class(model, test_loader, device)
    print(f"Nombre d'erreurs - Benign: {errors['benign']}, Malign: {errors['malign']}")


  

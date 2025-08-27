import lightning as L
from model_lightning import MyModel
from dataloader import get_loaders  # adapte le nom si besoin
import torch

# -- Paramètres fictifs, adapte selon ta config --
class ModelOpts:
    def __init__(self, name):
        self.name = name

class LossOpts:
    def __init__(self, name):
        self.name = name

class TrainPar:
    def __init__(self):
        self.eval_threshold = 0.5
        self.loss_opts = LossOpts("BCELogitsLoss")  # ou "FocalLoss", "CrossEntropyLoss"
        self.lr = 1e-4

# -- Prépare les loaders --
train_loader, val_loader, test_loader = get_loaders(batch_size=16, shuffle=True)

# -- Instancie le modèle --
model_opts = ModelOpts(name="resnet")  # ou "mobilenet", "inception", etc.
train_par = TrainPar()
model = MyModel(model_opts, train_par)

# -- Trainer Lightning --
trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",  # "gpu" si tu veux forcer le GPU
    devices=1,
    log_every_n_steps=10,
    default_root_dir="lightning_logs"
)

# -- Entraînement --
trainer.fit(model, train_loader, val_loader)

# -- Test final --
trainer.test(model, test_loader)
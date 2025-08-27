import lightning as L
from model_lightning import MyModel
from dataloader import get_loaders_transform
from lightning.pytorch.callbacks import EarlyStopping

class ModelOpts:
    def __init__(self, name):
        self.name = name

class LossOpts:
    def __init__(self, name):
        self.name = name

class TrainPar:
    def __init__(self, loss_name):
        self.eval_threshold = 0.5
        self.loss_opts = LossOpts(loss_name)
        self.lr = 1e-4

import lightning as L
from model_lightning import MyModel
from dataloader import get_loaders_transform

class ModelOpts:
    def __init__(self, name):
        self.name = name

class LossOpts:
    def __init__(self, name):
        self.name = name

class TrainPar:
    def __init__(self, loss_name):
        self.eval_threshold = 0.5
        self.loss_opts = LossOpts(loss_name)
        self.lr = 1e-4

# Utilisation de DenseNet et CrossEntropyLoss
model_opts = ModelOpts(name="mvsadensenet")
train_par = TrainPar("CrossEntropyLoss")

train_loader, val_loader, test_loader = get_loaders_transform(batch_size=16, shuffle=True)
model = MyModel(model_opts, train_par)

# Early stopping sur l'accuracy de validation
early_stop = EarlyStopping(
    monitor="val_accuracy",   # métrique à surveiller
    patience=10,              # nombre d'epochs sans amélioration avant arrêt
    mode="max",               # "max" pour accuracy
    verbose=True
)

trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
    default_root_dir="lightning_logs_densenet_CE",
    callbacks=[early_stop]    # <-- Ajout ici
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
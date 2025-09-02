import pytorch_lightning as L
from LateFusionLightning import LateFusionLightning
from dataloader import get_dataloaders, count_errors_by_class
from lightning.pytorch.callbacks import EarlyStopping

class ModelOpts:
    def __init__(self, name, eval_threshold=0.5):
        self.name = name
        self.eval_threshold = eval_threshold

class LossOpts:
    def __init__(self, name):
        self.name = name

class TrainPar:
    def __init__(self, loss_name):
        self.eval_threshold = 0.5
        self.loss_opts = LossOpts(loss_name)
        self.lr = 1e-4



model_opts = ModelOpts(name="latefusion_mvsadensenet") # Options du modèle: "latefusion_inception", "latefusion_densenet", "latefusion_mvsa", "latefusion_mvsadensenet"
train_par = TrainPar("BinaryCrossEntropyLoss")  # ou "BCELogitsLoss", "FocalLoss" selon ton besoin

model = LateFusionLightning(model_opts, train_par)

train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, shuffle=True)

early_stop = EarlyStopping(
    monitor="val_accuracy",   # métrique à surveiller
    patience=10,              # nombre d'epochs sans amélioration avant arrêt
    mode="max",               # "max" pour accuracy
    verbose=True
)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
    default_root_dir="lightning_logs_latefusion"
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinarySpecificity
from collections import defaultdict
from loss import BCELogitsLoss, FocalLoss
from models import LateFusionModel  # adapte l'import si besoin

class LateFusionLightning(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        self.model_opts = model_opts
        self.train_par = train_par

        self.model = LateFusionModel(num_tabular_features=1)  # adapte selon ton modèle

        self.eval_threshold = train_par.eval_threshold
        self.loss_name = train_par.loss_opts.name

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

        self.train_specificity = BinarySpecificity()
        self.val_specificity = BinarySpecificity()
        self.test_specificity = BinarySpecificity()

    def forward(self, x_img, x_tab):
        return self.model(x_img, x_tab)

    def get_loss(self, pred, label):
        if self.loss_name == "BCELogitsLoss":
            return BCELogitsLoss(pred, label)
#Combine un BCE Binary cross entropy et un sigmoid - stable pour l'entrainement
        elif self.loss_name == "FocalLoss":
            return FocalLoss(pred, label)
        else:
            raise ValueError(f"Loss {self.loss_name} is not supported")
#Accorde plus d'importance aux erreurs de classification 
    def training_step(self, batch, batch_idx):
        img, mask, label, tab = batch
        pred = self(img, tab)
        probs = torch.sigmoid(pred)  # Convert logits to probabilities
        loss = self.get_loss(pred, label)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy(probs, label), batch_size=label.size(0))
        self.log('train_recall', self.train_recall(probs, label), batch_size=label.size(0))
        self.log('train_specificity', self.train_specificity(probs, label), batch_size=label.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, label, tab = batch
        pred = self(img, tab)
        probs = torch.sigmoid(pred)
        loss = self.get_loss(pred, label)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy(probs, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall(probs, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_specificity', self.val_specificity(probs, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        pass  # Plus de logique patient à ce niveau

    def test_step(self, batch, batch_idx):
        img, mask, label, tab = batch
        pred = self(img, tab)
        probs = torch.sigmoid(pred)
        loss = self.get_loss(pred, label)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy(probs, label), batch_size=label.size(0))
        self.log('test_recall', self.test_recall(probs, label), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(probs, label), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(probs, label), batch_size=label.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_par.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


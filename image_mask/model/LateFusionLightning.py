import pytorch_lightning as L
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinarySpecificity
from collections import defaultdict
from loss import BCELogitsLoss, FocalLoss, BinaryCrossEntropyLoss
from LF_models import LateFusionModel_inceptiont, LateFusionModel_densenet, LateFusionModel_CustomMVSANet, LateFusionModel_CustomMVSADenseNet


class LateFusionLightning(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        self.model_opts = model_opts
        self.train_par = train_par
        self.all_test_preds = []
        self.all_test_labels = []   

        # Sélection du modèle selon le nom

        if model_opts.name == "latefusion_inception":
            self.model = LateFusionModel_inceptiont(num_tabular_features=1)
        elif model_opts.name == "latefusion_densenet":
            self.model = LateFusionModel_densenet(num_tabular_features=1)
        elif model_opts.name == "latefusion_mvsa":
            self.model = LateFusionModel_CustomMVSANet(num_tabular_features=1)
        elif model_opts.name == "latefusion_mvsadensenet":
            self.model = LateFusionModel_CustomMVSADenseNet(num_tabular_features=1)
        else:
            raise ValueError(f"Modelo non supporté: {model_opts.name}")


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
        elif self.loss_name == "FocalLoss":
            return FocalLoss(pred, label)
        elif self.loss_name == "BinaryCrossEntropyLoss":
            # Si pred.shape = [batch_size, 2], prendre la colonne 1 (proba positive)
            if pred.shape[1] == 2:
                pred = torch.softmax(pred, dim=1)[:, 1]
            label = label.float()
            return BinaryCrossEntropyLoss(pred, label)
        else:
            raise ValueError(f"Loss {self.loss_name} is not supported")
        
#Accorde plus d'importance aux erreurs de classification 
    def training_step(self, batch, batch_idx):
        img, mask, label, sphericity = batch
        tab = sphericity.float()
        pred = self(img, tab)
        loss = self.get_loss(pred, label)

        if pred.shape[1] == 2:
            probs = torch.softmax(pred, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(pred).squeeze()

        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy(probs, label), batch_size=label.size(0))
        self.log('train_recall', self.train_recall(probs, label), batch_size=label.size(0))
        self.log('train_specificity', self.train_specificity(probs, label), batch_size=label.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, label, sphericity = batch
        tab = sphericity.float()
        pred = self(img, tab)
        loss = self.get_loss(pred, label)

        if pred.shape[1] == 2:
            probs = torch.softmax(pred, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(pred).squeeze()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy(probs, label.float()), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall(probs, label.float()), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_specificity', self.val_specificity(probs, label.float()), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, mask, label, sphericity = batch
        tab = sphericity.float()
        pred = self(img, tab)

        if pred.shape[1] == 2:
            probs = torch.softmax(pred, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(pred).squeeze()

        # Binarise selon le threshold
        preds_bin = (probs > self.eval_threshold).int().cpu().numpy()
        labels_bin = label.int().cpu().numpy()

        self.all_test_preds.extend(preds_bin.tolist())
        self.all_test_labels.extend(labels_bin.tolist())

        loss = self.get_loss(pred, label)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy(probs, label), batch_size=label.size(0))
        self.log('test_recall', self.test_recall(probs, label), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(probs, label), batch_size=label.size(0))
        return loss

    def on_test_epoch_end(self):
        y_true = np.array(self.all_test_labels)
        y_pred = np.array(self.all_test_preds)
        cm = confusion_matrix(y_true, y_pred)
        print("Matrice de confusion :")
        print(cm)

        # Calcul des métriques à partir de la matrice de confusion
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            print(f"Test Accuracy (from confusion matrix): {accuracy:.4f}")
            print(f"Test Recall (from confusion matrix): {recall:.4f}")
            print(f"Test Specificity (from confusion matrix): {specificity:.4f}")
        else:
            print("Attention : la matrice de confusion n'est pas 2x2, vérifie le nombre de classes.")

        # Nettoyage pour les prochains tests
        self.all_test_preds.clear()
        self.all_test_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_par.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


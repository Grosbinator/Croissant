import lightning as L
import torch
from torch import nn
import torchmetrics
import numpy as np
from loss import BCELogitsLoss, FocalLossBinary, ComboLoss  # Update import with folder name
from models import CustomResnet, CustomMobileNet, Custominceptiont, Custom_densenet, Custom_vgg16, CustomResnet101, CustomMVSANet, CustomMVSADenseNet  # Update import with folder name
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassSpecificity
from collections import defaultdict
from confusion_matrix import plot_confusion_matrix

class MyModel(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        self.loss_name = train_par.loss_opts.name 
        self.model_opts = model_opts
        self.train_par = train_par
        self.test_confmat = torchmetrics.ConfusionMatrix(task="binary",num_classes=2)

        if self.loss_name == "CrossEntropyLoss":
            out_features = 2
        else:
            out_features = 1
        
        # Modelo definido dinámicamente basado en `model_opts.name`
        if model_opts.name == "resnet":
            self.model = CustomResnet(out_features=out_features)
        elif model_opts.name == "resnet101":
            self.model = CustomResnet101(out_features=out_features)
        elif model_opts.name == "mobilenet":
            self.model = CustomMobileNet(out_features=out_features)
        elif model_opts.name == "inception":
            self.model = Custominceptiont(out_features=out_features)
        elif model_opts.name == "densenet":
            self.model = Custom_densenet(out_features=out_features)
        elif model_opts.name == "vgg16":
            self.model = Custom_vgg16(out_features=out_features)
        elif model_opts.name == "mvsa":
            self.model = CustomMVSANet(out_features=out_features)
        elif model_opts.name == "mvsadensenet":
            self.model = CustomMVSADenseNet(out_features=out_features)
        else:
            raise ValueError(f"Modelo no soportado: {model_opts.name}")

        self.eval_threshold = train_par.eval_threshold
        self.loss_name = train_par.loss_opts.name  # Nombre de la pérdida (e.g., "BCELogitsLoss" o "FocalLoss")

        # Métricas para clasificación multicategoría
        self.train_accuracy = MulticlassAccuracy(num_classes=2)
        self.val_accuracy = MulticlassAccuracy(num_classes=2)
        self.test_accuracy = MulticlassAccuracy(num_classes=2)

        # Sensibilidad y especificidad
        self.train_recall = MulticlassRecall(num_classes=2, average="macro")
        self.val_recall = MulticlassRecall(num_classes=2, average="macro")
        self.test_recall = MulticlassRecall(num_classes=2, average="macro")

        self.train_specificity = MulticlassSpecificity(num_classes=2, average="macro")
        self.val_specificity = MulticlassSpecificity(num_classes=2, average="macro")
        self.test_specificity = MulticlassSpecificity(num_classes=2, average="macro")

        self.val_patient_predictions = defaultdict(list)
        self.val_patient_labels = {}

        self.test_patient_predictions = defaultdict(list)
        self.test_patient_labels = {}

    def forward(self, x):
        return self.model(x)

    def get_loss(self, pred, label):
        if self.loss_name == "BCELogitsLoss":
            return BCELogitsLoss(pred, label)
        elif self.loss_name == "FocalLoss":
            return FocalLossBinary(pred, label)
        elif self.loss_name == "CrossEntropyLoss":
            # Pour CrossEntropy, pas de unsqueeze ni float sur label
            from loss import CrossEntropyLoss
            # pred: [batch, 2], label: [batch] (long)
            class_weights = torch.FloatTensor([0.9, 1.5]).to(pred.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            return loss_fn(pred, label)
        elif self.loss_name == "ComboLoss":
            from loss import ComboLoss
            loss_fn = ComboLoss(ce_weight=0.5, focal_weight=0.5)  # tu peux ajuster les poids
            return loss_fn(pred, label)
        else:
            raise ValueError(f"Loss {self.loss_name} is not supported")
        

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.loss_name == "CrossEntropyLoss":
            # label doit être [batch] (long)
            pred = self(img)
            loss = self.get_loss(pred, label.long())
            pred_class = pred.argmax(dim=1)
            label_class = label.long()
        else:
            # BCE ou Focal : label doit être [batch, 1] (float)
            label = label.unsqueeze(1).float()
            pred = self(img)
            loss = self.get_loss(pred, label)
            pred_class = (pred > 0).long().squeeze(1)
            label_class = label.long().squeeze(1)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy(pred_class, label_class), batch_size=label.size(0))
        self.log('train_recall', self.train_recall(pred_class, label_class), batch_size=label.size(0))
        self.log('train_specificity', self.train_specificity(pred_class, label_class), batch_size=label.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        if self.loss_name == "CrossEntropyLoss":
            pred = self(img)
            loss = self.get_loss(pred, label.long())
            pred_class = pred.argmax(dim=1)
            label_class = label.long()
        else:
            label = label.unsqueeze(1).float()
            pred = self(img)
            loss = self.get_loss(pred, label)
            pred_class = (pred > 0).long().squeeze(1)
            label_class = label.long().squeeze(1)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy(pred_class, label_class), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall(pred_class, label_class), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_specificity', self.val_specificity(pred_class, label_class), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        if self.loss_name == "CrossEntropyLoss":
            pred = self(img)
            loss = self.get_loss(pred, label.long())
            pred_class = pred.argmax(dim=1)
            label_class = label.long()
        else:
            label = label.unsqueeze(1).float()
            pred = self(img)
            loss = self.get_loss(pred, label)
            pred_class = (pred > 0).long().squeeze(1)
            label_class = label.long().squeeze(1)

        self.test_confmat.update(pred_class, label_class)  # <-- AJOUTE CETTE LIGNE

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy(pred_class, label_class), batch_size=label.size(0))
        self.log('test_recall', self.test_recall(pred_class, label_class), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(pred_class, label_class), batch_size=label.size(0))
        return loss
    
    def on_test_epoch_end(self):
        confmat = self.test_confmat.compute().cpu().numpy()
        print("\nMatrice de confusion (torchmetrics):\n", confmat)
        self.test_confmat.reset()
        class_names = ["malignant", "benign"]  # adapte selon tes classes
        plot_confusion_matrix(
        confmat,
        class_names,
        save_path=f"confmat_{self.model_opts.name}.png",
        title=f"Matrice de confusion : {self.model_opts.name}"
        )

        # Pour 2 classes : confmat = [[TN, FP], [FN, TP]]
        TN, FP = confmat[0, 0], confmat[0, 1]
        FN, TP = confmat[1, 0], confmat[1, 1]
        total = TN + FP + FN + TP

        accuracy = (TP + TN) / total if total > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"Accuracy (global):    {accuracy:.4f}")
        print(f"Recall (sensibilité): {recall:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_par.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class MyModelMulticlassAlt(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        self.model_opts = model_opts
        self.train_par = train_par

        # Modelo definido dinámicamente basado en `model_opts.name`
        if model_opts.name == "resnet":
            self.model = CustomResnet(out_features=3)  # Cambiar a 3 clases
        elif model_opts.name == "mobilenet":
            self.model = CustomMobileNet(out_features=3)
        elif model_opts.name == "inception":
            self.model = Custominceptiont(out_features=3)
        elif model_opts.name == "densenet":
            self.model = Custom_densenet(out_features=3)
        elif model_opts.name == "vgg16":
            self.model = Custom_vgg16(out_features=3)
        else:
            raise ValueError(f"Modelo no soportado: {model_opts.name}")

        self.eval_threshold = train_par.eval_threshold
        self.loss_name = train_par.loss_opts.name

        # Métricas para clasificación multicategoría
        self.train_accuracy = MulticlassAccuracy(num_classes=3, average="macro")
        self.val_accuracy = MulticlassAccuracy(num_classes=3, average="macro")
        self.test_accuracy = MulticlassAccuracy(num_classes=3, average="macro")

        self.train_recall = MulticlassRecall(num_classes=3, average="macro")
        self.val_recall = MulticlassRecall(num_classes=3, average="macro")
        self.test_recall = MulticlassRecall(num_classes=3, average="macro")

        self.train_specificity = MulticlassSpecificity(num_classes=3, average="macro")
        self.val_specificity = MulticlassSpecificity(num_classes=3, average="macro")
        self.test_specificity = MulticlassSpecificity(num_classes=3, average="macro")

        self.val_patient_predictions = defaultdict(list)
        self.val_patient_labels = {}

        self.test_patient_predictions = defaultdict(list)
        self.test_patient_labels = {}

    def forward(self, x):
        return self.model(x)

    def get_loss(self, pred, label):
        class_weights = torch.FloatTensor([1.0, 1.0, 0.8]).cuda()  # Ajusta los pesos según la distribución
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        return loss_fn(pred, label)

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self(img)
        loss = self.get_loss(pred, label)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy(pred, label), batch_size=label.size(0))
        self.log('train_recall', self.train_recall(pred, label), batch_size=label.size(0))
        self.log('train_specificity', self.train_specificity(pred, label), batch_size=label.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, patient_ids = batch
        pred = self(img)
        loss = self.get_loss(pred, label)

        for i, patient_id in enumerate(patient_ids):
            self.val_patient_predictions[patient_id].append(pred[i].softmax(dim=-1).cpu().numpy())
            self.val_patient_labels[patient_id] = label[i].item()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_specificity', self.val_specificity(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        patient_final_predictions = {}
        for patient_id, preds in self.val_patient_predictions.items():
            avg_preds = np.mean(preds, axis=0)
            patient_final_predictions[patient_id] = np.argmax(avg_preds)

        y_true = list(self.val_patient_labels.values())
        y_pred = [patient_final_predictions[pid] for pid in self.val_patient_labels.keys()]

        self.log('val_patient_accuracy', np.mean([y_t == y_p for y_t, y_p in zip(y_true, y_pred)]), prog_bar=True, on_epoch=True)

        self.val_patient_predictions.clear()
        self.val_patient_labels.clear()

    def test_step(self, batch, batch_idx):
        img, label, patient_ids = batch
        pred = self(img)
        loss = self.get_loss(pred, label)

        for i, patient_id in enumerate(patient_ids):
            self.test_patient_predictions[patient_id].append(pred[i].softmax(dim=-1).cpu().numpy())
            self.test_patient_labels[patient_id] = label[i].item()

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy(pred, label), batch_size=label.size(0))
        self.log('test_recall', self.test_recall(pred, label), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(pred, label), batch_size=label.size(0))

    def on_test_epoch_end(self):
        patient_final_predictions = {}
        for patient_id, preds in self.test_patient_predictions.items():
            avg_preds = np.mean(preds, axis=0)
            patient_final_predictions[patient_id] = np.argmax(avg_preds)

        y_true = list(self.test_patient_labels.values())
        y_pred = [patient_final_predictions[pid] for pid in self.test_patient_labels.keys()]

        self.log('test_patient_accuracy', np.mean([y_t == y_p for y_t, y_p in zip(y_true, y_pred)]), prog_bar=True, on_epoch=True)

        self.test_patient_predictions.clear()
        self.test_patient_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_par.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class MyModelMulticlass(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        self.model_opts = model_opts
        self.train_par = train_par

        # Modelo definido dinámicamente basado en `model_opts.name`
        if model_opts.name == "resnet":
            self.model = CustomResnet(out_features=3)  # Cambiar a 3 clases
        elif model_opts.name == "mobilenet":
            self.model = CustomMobileNet(out_features=3)
        elif model_opts.name == "inception":
            self.model = Custominceptiont(out_features=3)
        elif model_opts.name == "densenet":
            self.model = Custom_densenet(out_features=3)
        elif model_opts.name == "vgg16":
            self.model = Custom_vgg16(out_features=3)
        else:
            raise ValueError(f"Modelo no soportado: {model_opts.name}")

        self.eval_threshold = train_par.eval_threshold
        self.loss_name = train_par.loss_opts.name

        # Métricas para clasificación multicategoría
        self.train_accuracy = MulticlassAccuracy(num_classes=3, average="macro")
        self.val_accuracy = MulticlassAccuracy(num_classes=3, average="macro")
        self.test_accuracy = MulticlassAccuracy(num_classes=3, average="macro")

        self.train_recall = MulticlassRecall(num_classes=3, average="macro")
        self.val_recall = MulticlassRecall(num_classes=3, average="macro")
        self.test_recall = MulticlassRecall(num_classes=3, average="macro")

        self.train_specificity = MulticlassSpecificity(num_classes=3, average="macro")
        self.val_specificity = MulticlassSpecificity(num_classes=3, average="macro")
        self.test_specificity = MulticlassSpecificity(num_classes=3, average="macro")

    def forward(self, x):
        return self.model(x)

    def get_loss(self, pred, label):
        class_weights = torch.FloatTensor([1.0, 1.0, 0.8]).cuda()  # Ajusta los pesos según la distribución
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        return loss_fn(pred, label)

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self(img)
        loss = self.get_loss(pred, label)

        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy(pred, label), batch_size=label.size(0))
        self.log('train_recall', self.train_recall(pred, label), batch_size=label.size(0))
        self.log('train_specificity', self.train_specificity(pred, label), batch_size=label.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        img, label,_ = batch
        pred = self(img)
        loss = self.get_loss(pred, label)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)
        self.log('val_specificity', self.val_specificity(pred, label), batch_size=label.size(0), prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        pred = self(img)
        loss = self.get_loss(pred, label)

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy(pred, label), batch_size=label.size(0))
        self.log('test_recall', self.test_recall(pred, label), batch_size=label.size(0))
        self.log('test_specificity', self.test_specificity(pred, label), batch_size=label.size(0))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_par.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
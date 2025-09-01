import torch
import torch.nn as nn
import torchvision.models as models
from dataloader import MaskDataset, get_dataloaders
import pandas as pd

class CustomResnet(nn.Module):
    def __init__(self, out_features = 2):
        super(CustomResnet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)  # 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Utiliza solo 'self.features' para pasar los datos a través de ResNet
        x = self.features(x)

        # Aplana la salida para la capa lineal
        x = x.view(x.size(0), -1)

        # Pasa los datos a través de las capas adicionales
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



# Define the model
class CustomMobileNet(nn.Module):
    def __init__(self, out_features=2):
        super(CustomMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=1)  # Remove or comment out

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Return raw logits
        x = self.fc3(x)

        # No softmax here. CrossEntropyLoss will handle the softmax internally.
        return x


class Custominceptiont(nn.Module):
    def __init__(self, out_features=2):
        super(Custominceptiont, self).__init__()
        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()  # Quita la última capa
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        # Verifica el tamaño de entrada y redimensiona si es necesario
        if x.shape[2] < 75 or x.shape[3] < 75:
            print(f"Resizing input from {x.shape[2:]} to (75, 75)")
            #x = F.interpolate(x, size=(75, 75), mode="bilinear", align_corners=False)

        if self.training:
            x, _ = self.base_model(x)
        else:
            x = self.base_model(x)

        x = x.view(x.size(0), -1)  # Aplanar
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x




# Define the model
class Custom_densenet(nn.Module):
    def __init__(self, out_features=2):
        super(Custom_densenet, self).__init__()
        self.base_model = models.densenet121(pretrained=True).features  # 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc1 = nn.Linear(1024, 128)  # 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        # x = self.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



# Define the model
class Custom_vgg16(nn.Module):
    def __init__(self, out_features=2):
        super(Custom_vgg16, self).__init__()
        self.base_model = models.vgg16(pretrained=True).features  # 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc1 = nn.Linear(512, 128)  # 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        #x = self.softmax(self.fc3(x))
        x = self.fc3(x)

        return x
    
    
class PyTorchCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), 
        )

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), 
        )
        
        self.fc_layers = torch.nn.Sequential(
            # hidden layer
            torch.nn.Linear(39200, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # output layer
            torch.nn.Linear(20, num_classes)
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.block1(x)
        # print(f"Shape after block1: {x.shape}")
        x = self.block2(x)
        x = self.block3(x)
        # print(x.shape)
        # print(f"Shape after CNN layers: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        #print("Flatten shape:", x.shape)
        logits = self.fc_layers(x)
        return logits
    


class LateFusionModel(nn.Module):
    def __init__(self, num_tabular_features=1):
        super().__init__()
        self.cnn = Custominceptiont() 
        self.cnn.fc = nn.Identity() #Enlève la couche finale de classification
        cnn_feature_size = 2
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(cnn_feature_size + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, sphericity):
        x_img = self.cnn(image)
        x_tab = self.tabular(sphericity)
        if x_tab.dim() > 2:
            x_tab = x_tab.view(x_tab.size(0), -1)
        x = torch.cat([x_img, x_tab], dim=1)
        return self.classifier(x).squeeze(-1)
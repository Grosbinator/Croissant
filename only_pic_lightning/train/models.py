import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd

class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class CustomMobileNet(nn.Module):
    def __init__(self):
        super(CustomMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Custominceptiont(nn.Module):
    def __init__(self):
        super(Custominceptiont, self).__init__()
        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        if x.shape[2] < 75 or x.shape[3] < 75:
            print(f"Resizing input from {x.shape[2:]} to (75, 75)")
            # x = F.interpolate(x, size=(75, 75), mode="bilinear", align_corners=False)
        if self.training:
            x, _ = self.base_model(x)
        else:
            x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Custom_densenet(nn.Module):
    def __init__(self):
        super(Custom_densenet, self).__init__()
        self.base_model = models.densenet121(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Custom_vgg16(nn.Module):
    def __init__(self):
        super(Custom_vgg16, self).__init__()
        self.base_model = models.vgg16(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class CustomMVSANet(nn.Module):
    def __init__(self, out_features=1):
        super(CustomMVSANet, self).__init__()
        # Exemple : on utilise resnet50 comme backbone et on ajoute une couche d'attention simple
        self.base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        attn = self.attention(x)
        x = x * attn  # MVSA: attention multiplicative
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
     
class CustomMVSADenseNet(nn.Module):
    def __init__(self, out_features=2):
        super(CustomMVSADenseNet, self).__init__()
        # Utilise DenseNet121 comme backbone
        self.base_model = models.densenet121(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 128)
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        attn = self.attention(x)
        x = x * attn  # MVSA: attention multiplicative
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
class CustomResnet101(nn.Module):
    def __init__(self):
        super(CustomResnet101, self).__init__()
        self.base_model = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
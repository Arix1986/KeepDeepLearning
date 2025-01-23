from torch import nn 
from torchvision.models import  efficientnet_b0, EfficientNet_B0_Weights
import torch


class NetModel(nn.Module):
    def __init__(self, activation_function, num_features_tab):
        super(NetModel, self).__init__()
        self.activation = activation_function
        
       
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
       
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
       
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )
        
      
        self.fc0 = nn.Sequential(
            nn.Linear(num_features_tab, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.3)
        )
        
       
        self.combined_fc = nn.Sequential(
            nn.Linear(256 * 16 * 16 + 64, 128), 
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(0.4)
        )
        
       
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Linear(64, 10)  

    def forward(self, x_img, x_tab):
       
        ximg = self.layer1(x_img)
        ximg = self.layer2(ximg)
        ximg = self.layer3(ximg)
        ximg = ximg.view(ximg.size(0), -1)  
        
      
        xtab = self.fc0(x_tab)

      
        x_combined = torch.cat((ximg, xtab), dim=1)
        x_combined = self.combined_fc(x_combined)
        x_combined = self.fc2(x_combined)
        x_combined = self.fc3(x_combined)

        return x_combined
      
        
        
    
    
    
    
class TransferLearning(nn.Module):
    
    def __init__(self, activation_function,num_features_tab):
        super(TransferLearning,self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in list(self.efficientnet.parameters())[:-5]:
            param.requires_grad = False
        
        self.activation= activation_function
        
        self.feature_extractor = nn.Sequential(
            self.efficientnet.features,  
            self.efficientnet.avgpool, 
            nn.Flatten()               
        )
        self.fc0=nn.Sequential(
            nn.Linear(num_features_tab, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.3),
           
           
        )
        self.fc1=nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(0.4)
           
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(1280 + 32, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64,32),  
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(0.4)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            self.activation,
            nn.Dropout(0.4)
        )
        self.fc4 = nn.Linear(16, 10)
        
    def forward(self,x_img,x_tab):
        with torch.no_grad():
            ximg=self.feature_extractor(x_img)
        ximg = ximg.view(x_img.size(0), -1)
        xtab = self.fc0(x_tab)
        xtab = self.fc1(xtab)

       
        x_combined = torch.cat((ximg, xtab), dim=1)
        x_combined = self.combined_fc(x_combined)
        x_combined = self.fc2(x_combined)
        x_combined = self.fc3(x_combined)
        x_combined = self.fc4(x_combined)
        return x_combined


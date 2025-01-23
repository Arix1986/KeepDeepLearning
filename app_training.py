import pickle
from torch import nn 
import torch
import torch.optim as optim
from tqdm import tqdm
from app_model import NetModel, TransferLearning


   
class Training():
    def __init__(self,model,activation_function,lr,weight_d,num_features_tab,num_epoch=30, train_loader=None, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        list_models={'A':NetModel(activation_function,num_features_tab).to(self.device),'B':TransferLearning(activation_function,num_features_tab).to(self.device)}
        self.type_model=model
        self.model= list_models[model]
        self.criterion=nn.CrossEntropyLoss() 
        self.optimizer=optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_d)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy=[]
        self.val_accuracy=[]
        
        print("[INFO]: Entrenando red neuronal...")
    
    def run(self):
        best_val_loss = float('inf') 
        best_model_state = None      
        pbar = tqdm(range(self.num_epoch), desc='Training', unit='epoch',
                    postfix={'train_loss': 0.0, 'val_loss': 0.0,'train_accuracy': 0.0 , 'val_accuracy':0.0})
        for epoch in pbar:
            self.model.train()
            avg_loss = 0.0
            t_accuracy=0.0
            for (images, tabular_data), labels in self.train_loader:
                images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)

                output = self.model(images, tabular_data)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() / len(self.train_loader)
                t_accuracy += (output.argmax(dim=1) == labels).sum().item() / labels.size(0)
            val_loss = 0.0
            v_accuracy=0.0
            t_accuracy /= len(self.train_loader)
            self.model.eval()
            with torch.no_grad():
                for (images, tabular_data), labels in self.val_loader:
                    images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)

                    test_output = self.model(images, tabular_data)
                    val_loss += self.criterion(test_output, labels).item() / len(self.val_loader)
                    v_accuracy += (test_output.argmax(dim=1) == labels).sum().item() / labels.size(0)
            v_accuracy /= len(self.val_loader)
            self.train_losses.append(avg_loss)
            self.val_losses.append(val_loss)
            self.train_accuracy.append(t_accuracy)
            self.val_accuracy.append(v_accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
            
            pbar.set_postfix(train_loss=avg_loss, val_loss=val_loss,train_accuracy=t_accuracy,val_accuracy=v_accuracy)  
        model_path = f'./modelos/best_model_{best_val_loss}_{self.type_model}.pth'
        torch.save(best_model_state, model_path)
        
import optuna
import torch
import torch.nn as nn
from app_training import Training


class HyperparameterOptimization:
    def __init__(self, train_loader, val_loader,num_features_tab,name_study='hyperparameter_optimization',model='A', num_trials=40, num_epochs=35, device=None):
        self.train_loader = train_loader
        self.model=model
        self.val_loader = val_loader
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.name_study= name_study
        self.num_features_tab=num_features_tab
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def objective(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        self.activation_name = trial.suggest_categorical("activation_function", ["ReLU","PReLU"])
        self.activation_function = {"ReLU": nn.ReLU(), "PReLU": nn.PReLU()}.get(self.activation_name)
        trainer = Training(
            model=self.model,
            activation_function=self.activation_function,
            lr=lr,
            weight_d=weight_decay,
            num_features_tab=self.num_features_tab,
            num_epoch=self.num_epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader
        )
        trainer.run()
        val_losses= trainer.val_losses
      
        return val_losses[-1]  
    def optimize(self):
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna_study.db", 
            study_name=self.name_study, 
            load_if_exists=True  
        )
        study.optimize(self.objective, n_trials=self.num_trials)
        return study


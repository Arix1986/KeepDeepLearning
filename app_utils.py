

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image


def display_loss_and_accuracy(loss_epoch_tr, loss_epoch_val, acc_epoch_tr, acc_epoch_val,  n_epochs):
    """
    Función que muestra un gráfico con dos subplots:
    - El primer subplot muestra el loss por epoch
    - El segundo subplot muestra la precisión por epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(n_epochs), loss_epoch_tr, label='Entrenamiento')
    ax1.plot(range(n_epochs), loss_epoch_val, label='Validación')
    ax1.legend(loc='upper right')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Loss')

    ax2.plot(range(n_epochs), acc_epoch_tr, label='Entrenamiento')
    ax2.plot(range(n_epochs), acc_epoch_val, label='Validación')
    #ax2.axhline(y=test_acc, color='red', linestyle='--', label='Test')
    ax2.legend(loc='lower right')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch #')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()
    
def linear_feature_importance(model, column_names):
    """
    Extrae la importancia de características directamente desde una capa lineal del modelo.

    Args:
        model: Modelo entrenado con capas lineales.
        column_names: Lista de nombres de las características tabulares.

    Returns:
        Visualización de importancia de características.
    """
    weights_fc0 = model.fc0[0].weight.detach().cpu().numpy()
    weights_fc1 = model.fc1[0].weight.detach().cpu().numpy()
    propagated_importance = np.abs(weights_fc1) @ np.abs(weights_fc0)
    feature_importance = propagated_importance.mean(axis=0)
    
    plt.barh(column_names, feature_importance)
    plt.xlabel("Feature Importance (Weights)")
    plt.title("Linear Feature Importance")
    plt.show()

  
    



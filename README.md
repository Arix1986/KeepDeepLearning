# KeepDeepLearning
MÓDULO DE DEEP LEARNING: ANÁLISIS DE DATOS TABULARES E IMÁGENES

# Proyecto de Análisis y Optimización

Este proyecto organiza los archivos y carpetas de la siguiente manera:

## Estructura del Proyecto

📂 **Root/**  
├── 📄 **app_features.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Análisis exploratorio de los datos.  
├── 📄 **app_dataset.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Implementa las clases `ProcessDataset` y `DataBuilder` para el preprocesamiento.  
├── 📄 **app_hyperparam.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Marco para la optimización de hiperparámetros con Optuna.  
├── 📄 **app_training.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Clase para entrenar modelos.  
├── 📄 **app_model.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contiene las clases de los modelos `NetModel` y `TransferLearning`.  
├── 📄 **app_main.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Maneja el entrenamiento y la optimización del modelo.  
├── 📄 **app_test.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Realiza evaluaciones y análisis sobre los resultados.  
├── 📄 **app_utils.py**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Funciones de soporte para diversas operaciones.  

📂 **modelos/**  
├── 📄 **best_model_*.pth**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Modelos entrenados con los mejores resultados.  
├── 📄 **scaler_minmax.pkl**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scaler MinMax para normalización de datos.  
├── 📄 **kmeans.pkl**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Modelo KMeans para clustering de geolocalización.  

📂 **data/**  
├── 📄 **poi_train.csv**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Datos de entrenamiento.  
├── 📄 **poi_test.csv**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Datos de prueba.  

📄 **optuna_study.db**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Base de datos SQLite con los estudios realizados con Optuna.  
📄 **categories_index.json**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Archivo JSON con los índices de las categorías utilizadas.  

---

### Notas
- **Modelos entrenados:** Los mejores modelos se guardan en la carpeta `modelos/`.
- **Datos:** Los conjuntos de datos se encuentran en la carpeta `data/`.
- **Hiperparámetros:** La optimización está centralizada en el archivo `app_hyperparam.py` usando Optuna.


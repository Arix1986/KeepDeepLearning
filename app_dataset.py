
import ast
import json
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import joblib
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Embedding
import os
os.environ["OMP_NUM_THREADS"] = "4"

class ProcessDataset(Dataset):
    def __init__(self, dataframe,is_train=False, transform=None):
        self.is_train=is_train
        self.dataframe = self.preprocess(dataframe)
        self.transform = transform
        self.num_features=None
        self.dfp=self.dataframe.drop(['Visits','Likes', 'Dislikes','Bookmarks', 'name', 'tags', 'main_image_path', 'id', 'shortDescription','categories','categories_idx','category_embeddings','locationLon','locationLat'], axis=1)
        print(self.dfp.columns)
        self.features = torch.tensor(self.dataframe.drop(['Visits','Likes', 'Dislikes','Bookmarks', 'name', 'tags', 'main_image_path', 'id', 'shortDescription','categories','categories_idx','category_embeddings','locationLon','locationLat'], axis=1).values, dtype=torch.float32)    
        total_interaction = (
            self.dataframe['Likes'].values + 
            self.dataframe['Bookmarks'].values + 
            self.dataframe['Dislikes'].values + 
            self.dataframe['Visits'].values
        )
        interaction_score = (
            (self.dataframe['Likes'].values + self.dataframe['Bookmarks'].values) / total_interaction
        )
        category_embeddings_tensor = torch.tensor(
            np.vstack(self.dataframe['category_embeddings'].values), 
            dtype=torch.float32
        )
        print(self.features.shape)
        self.features = torch.cat((self.features, category_embeddings_tensor), dim=1)
        print(self.features.shape)
        num_classes = 10
        bins = np.linspace(0, 1, num_classes + 1)
        labels = np.digitize(interaction_score, bins) - 1  
        labels = np.clip(labels, 0, num_classes - 1)
        self.labels= torch.tensor(labels, dtype=torch.long)
        self.num_features = self.features.shape[1]
       
        
    def preprocess(self, df):
        log_columns = ['Likes', 'Dislikes', 'Bookmarks']
        minmax_columns = ['Visits', 'xps', 'tier', 'geo_cluster']
        df[log_columns] = np.log1p(df[log_columns])
        embedding_dim = 8 
        if self.is_train:
            kmeans = KMeans(n_clusters=6, random_state=42,n_init=10)
            coordinates = df[['locationLon','locationLat']].values
            kmeans.fit(coordinates)
            geo_cluster=kmeans.predict(coordinates)
            df['geo_cluster']=geo_cluster
            with open('./modelos/kmeans_model.pkl', 'wb') as file:
                 pickle.dump(kmeans, file)
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            df[minmax_columns] = scaler.fit_transform(df[minmax_columns])
            joblib.dump(scaler, './modelos/minmax_scaler.pkl')   
            df['categories'] = df['categories'].apply(ast.literal_eval)
            df['categories']= df['categories'].apply(lambda x : sorted(x))
            df['categories']=df['categories'].apply(str)
            categories = df['categories'].unique().tolist()
            num_categories = len(categories)
            categories_idx={cat:idx for idx,cat in enumerate(categories)}
            df['categories_idx']=df['categories'].map(categories_idx)
            with open('category_to_index.json', 'w') as f:
                 json.dump(categories_idx, f)
             
        else:         
            with open('./modelos/kmeans_model.pkl', 'rb') as file:
                loaded_kmeans = pickle.load(file)
                geo_cluster = loaded_kmeans.predict(df[['locationLon', 'locationLat']].values)
                df['geo_cluster']=geo_cluster
                scaler = joblib.load('./modelos/minmax_scaler.pkl')
                df[minmax_columns] = scaler.transform(df[minmax_columns])
            with open('category_to_index.json', 'r') as f:
                 category_to_index = json.load(f) 
                 num_categories = len(category_to_index)
                 unknown_category_index = num_categories
                 df['categories_idx'] = df['categories'].map(lambda x: category_to_index.get(x, unknown_category_index)) 
                
        embedding_layer = Embedding(num_categories + 1, embedding_dim)
        category_embeddings = embedding_layer(torch.tensor(df['categories_idx'].values))
        df['category_embeddings'] = category_embeddings.tolist()         
        return df   

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['main_image_path']  
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        features_tensor = self.features[idx]
        labels_tensor = self.labels[idx]
       
        return (image, features_tensor), labels_tensor
    
    

class DataBuilder:
    def __init__(self, dataframe,test_size=0.2, seed=42, batch_size=32):
        
        self.dataframe = dataframe
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.set_seed(6)
    
    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

           

    def calculate_mean_std(self, image_paths):
        mean = np.zeros(3)
        std = np.zeros(3)
        num_images = len(image_paths)

        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)  
            mean += image.mean(dim=(1, 2)).numpy()
            std += image.std(dim=(1, 2)).numpy()

        mean /= num_images
        std /= num_images
        return mean, std

    def build(self):
        
        train_df, val_df = train_test_split(self.dataframe, test_size=self.test_size, random_state=self.seed)
        mean, std = self.calculate_mean_std(train_df['main_image_path'].tolist())
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = ProcessDataset(train_df,is_train=True, transform=transform)
        val_dataset = ProcessDataset(val_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, train_dataset.num_features   
    
    def build_test(self):
        
        mean, std = self.calculate_mean_std(self.dataframe['main_image_path'].tolist())
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
       
        test_dataset = ProcessDataset(self.dataframe, transform=transform)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
       
        
        return test_loader,  test_dataset.num_features
    
    
    
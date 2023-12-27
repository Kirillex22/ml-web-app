import pandas as pd
import numpy as np
import tensorflow as tf
import catboost as cb
import pickle

class ModelManager:
    
    def __init__(self):
        self.model = None
        with open('data/models/standart_scaler.pickle', 'rb') as file:
            self.sc = pickle.load(file)
        with open('data/models/pca.pickle', 'rb') as file:
            self.pca = pickle.load(file)
         
    def load(self, name, special_key = "pickle"):
        path = f'data/models/{name}'
        if special_key == "pickle":
            with open(path, 'rb') as f:
                self.model = pickle.load(f)  

        elif special_key == "catboost":
            from_file = cb.CatBoostRegressor()
            from_file.load_model(path) 
            self.model = from_file      

        elif special_key == "tensorflow":
            self.model = tf.keras.models.load_model(path) 

    def predict(self, X):     
        return self.model.predict(self.scale(X)).flatten()
    
    def transform_predict(self, X):
        return self.model.predict(
            self.pca.transform(self.scale(X))
        )
    
    def scale(self, X):
        return self.sc.transform(X)

        


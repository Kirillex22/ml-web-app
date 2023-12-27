import streamlit as st
import pandas as pd
import numpy as np
import statistics

import sys

sys.path.append('..')

from scripts.model_manager import ModelManager

fit_data = pd.read_csv('data/datasets/winequality.csv')
fit_data.drop(['quality'], axis = 1, inplace = True)
names = []
values = []

for label, content in fit_data.items():
    names.append(label)
    values.append(content)


def data_loader():
    data = st.file_uploader("Загрузите датасет в формате *.csv", type="csv")
    if data is not None:
        data = pd.read_csv(data)
    return data

def data_input(names, values):
    st.header("Введите данные")
    user_input = np.zeros(12)

    for i in range(12):
        user_input[i] = st.number_input(names[i], min_value=float(min(values[i])), max_value=float(max(values[i])), value=float(statistics.mean(values[i])))

    ch = st.checkbox('Подтвердить данные')
    if (ch):
        data = pd.DataFrame(user_input.reshape(1, -1), columns = names)
        return data

def predict_manager(data: pd.DataFrame):
    
    if ('quality' in data.columns.to_list()):
        data.drop(['quality'], axis = 1, inplace = True)

    st.markdown("Ваш датасет:")
    st.write(data[:5])
    mm = ModelManager()
         
    types = ["MLR с ElasticNet (sklearn)", "Ансамбли", "PCA + Ridge (sklearn)", "Глубокая НН (tensorflow)"]   
    models = ["Bagging (sklearn)", "GradientBoosting (catboost)", "Stacking(sklearn)"] 

    current_type = st.selectbox("Выберите тип модели", types, index=None)
               
    
    if current_type is not None:

        if current_type == types[0]:
            mm.load("mlr_elastic.pickle")
            st.write(mm.predict(data))

        elif current_type == types[1]:

            model_type = st.selectbox("Выберите тип модели", models, index=None)

            if model_type == models[0]:
                mm.load("bagging.pickle")
                st.write(mm.predict(data))
            elif model_type == models[1]:
                mm.load("catboost.cbm", special_key = "catboost")
                st.write(mm.predict(data))
            elif model_type == models[2]:
                mm.load("stacking.pickle")
                st.write(mm.predict(data))

        elif current_type == types[2]:
            mm.load("ridge_pca.pickle")
            st.write(mm.transform_predict(data))

        elif current_type == types[3]:
            mm.load("dnn.h5", special_key = "tensorflow")
            st.write(mm.predict(data))

    
st.title("Получение предсказаний от обученных моделей")

types = ["Загрузка датасета", "Ввод данных вручную"]
current_type = st.selectbox("Выберите тип предсказаний", types, index=None)

if current_type is not None:
    if current_type == types[0]:
        data = data_loader()          
    if current_type == types[1]:
        data = data_input(names, values)

    ch = st.checkbox('Перейти к моделям')
    if(ch):
        predict_manager(data)
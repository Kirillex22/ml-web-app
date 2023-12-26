import streamlit as st
import pandas as pd
import sys

sys.path.append('..')

from scripts.vis_handler import VisHandler

st.title("Визуализация")

data = pd.read_csv("data/datasets/winequality.csv")
handler = VisHandler(data)

types = ['Тепловая карта', 'Boxplot', 'Гистограмма', 'Диаграмма плотности распределения', 'Диаграмма рассеяния']
current_type = st.selectbox("Тип визуализации признаков", types)

if current_type is not None:
    if current_type == types[0]:
        handler.heatmap()
    elif current_type == types[1]:
        handler.boxplot()    
    elif current_type == types[2]:
        handler.hist()
    elif current_type == types[3]:
        handler.displot()
    elif current_type == types[4]:
        handler.lmplot()
    
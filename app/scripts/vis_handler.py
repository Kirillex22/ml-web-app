import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

class VisHandler:

    def __init__(self, data):
        self.data = data
        self.features = data.columns.to_list()


    def heatmap(self):
        st.title("Тепловая карта")
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            self.data.corr(), annot=True, cmap="YlGnBu", fmt=".2f"
        )
        st.pyplot(plt) 

            
    def boxplot(self):
        st.title("Boxplot")    
        ft = st.selectbox("Признак", self.features)
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x = self.data[ft], color="orange", linewidth = 1.5
        )
        st.pyplot(plt)


    def hist(self):
        st.title("Гистограмма")    
        ft = st.selectbox("Признак", self.features, key='selectbox1')
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[ft], bins=100, color = "green")
        st.pyplot(plt)


    def lmplot(self):
        st.title("Диаграмма рассеяния")    
        options = st.multiselect(
            'Признаки', self.features,
            ['quality', 'quality'], max_selections=2
        )
        plt.figure(figsize=(10, 6))
        try:
            sns.lmplot(x=options[0],y=options[1], data=self.data[:500])
        except:
            pass
        st.pyplot(plt)


    def displot(self):
        st.title("Диаграмма плотности распределения")    
        ft = st.selectbox("Выберите признак", self.features)
        plt.figure(figsize=(10, 6))
        sns.displot(data=self.data, x=ft, kind="kde")
        st.pyplot(plt)
        

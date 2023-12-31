import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Описание датасета")

st.markdown('## Тематика')
st.markdown("""
    **Оценка качества** красного и
    белого вина Винью-Верде с севера Португалии, данные разбиты на два файла, 
    соответствующих красному и белому винам.
""")

st.markdown('## Задача регрессии')
st.markdown("""
    Цель состоит в том, чтобы **предсказать оценку качества вина** 
    по шкале от 0 до 10 на основе физико-химических тестов.
    """)

st.markdown('## Описание признаков')

st.markdown("""
- *Fixed acidity*: данный тип кислот участвует 
в сбалансированности вкуса вина, привносит свежесть вкусу          
     """)

st.markdown("""
- *Volatile acidity*: наличие летучих кислот
в вине, например, таких как уксусная кислота
    """)

st.markdown("""
- *Citric acid*: cодержание лимонной кислоты
    """)

st.markdown("""
- *Residual sugar*: показывает количество
сахара, который не был превращен в спирт в процессе ферментации
 вина
    """)

st.markdown("""
- *Chlorides*: показывает cодержание хлоридов
    """)

st.markdown("""
- *Free/Total sulfur dioxide*:  показывает число сульфитов в свободном и связанном виде, которые используются
виноделии в качестве безопасного антисептика
    """)

st.markdown("""
- *Density*: показывает плотность напитка
    """)

st.markdown("""
- *pH*: показывает кислотность вина по водородной шкале (влияет на цвет)
    """)

st.markdown("""
- *Sulphates*:  количество сульфатов (их задача — подавить и убить нежелательные дрожжи и бактерии, 
чтобы вино не окислялось преждевременно, а также могло переносить транспортировку и выдержку)
    """)

st.markdown("""
- *Alcohol*: количество спирта (характеризует крепость вина)
    """)

st.markdown("""
- *оценка качества от 0 до 10*: cубъективная оценка качества вина
    """)

df_red = pd.read_csv("data/datasets/winequality-red.csv", sep = ';')
df_white = pd.read_csv("data/datasets/winequality-white.csv", sep = ';')
data = pd.read_csv("data/datasets/winequality.csv")

st.markdown("#### Датасет для красного вина до обработки")
st.write(df_red.head(10))
st.markdown("#### Датасет для белого вина до обработки")
st.write(df_white.head(10))


st.markdown("## Особенности предобработки")
st.markdown("**1. Объединение данных**")
st.markdown("""
        Изначально имелось два датасета: один для красного вина, другой - для белого. 
        В первую очередь возникла необходимость объединить их, дабы упростить дальнейшую работу
        с данными. Однако просто так сконкатенировать датасеты не получится, поскольку нельзя приравнивать
        красные и белые вина по отношению друг к другу. Решением данной проблемы стало добавление дополнительного 
        бинарного признака Color, где единица означает, что вино красное, а 0 - белое.      
        """)

st.code("""
    data_white['color'] = 0
    data_red['color'] = 1
    data = pd.concat([data_white, data_red], axis = 0)
    """)

st.markdown("В результате датасет стал выглядеть следующим образом:")
st.write(data.head(10))

st.markdown("**2. Отсутствие обработки пропущенных значений**")
st.markdown("""
        Поскольку исходные датасеты не содержали абсолютно никаких пропущенных значений, 
        полученный объединенный датасет также всюду непуст, поэтому данная обработка не требовалась.
        """)

st.markdown("**3. Отсутствие обработки выбросов**")
st.markdown("""
        Анализ гистограмм всех признаков показал, что данные выглядят чистыми и не требуют
        удаления каких-либо аномальных значений (выбросов). Многие из распределений, как, например, pH, оказались
        крайне близки к нормальным. Остальные же распределения не вызывают особых подозрений, поскольку выглядят 
        вполне правдоподобно и не характеризуются резкими скачками значений исследуемых величин. 
        """)



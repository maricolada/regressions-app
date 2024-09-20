import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Название
#Описание

st.title("Линейная регрессия и визуализация данных")
st.write("Здесь вы можете получить результат линейной регрессии для тренировочной или тестовой выборки, а также возможность построить scatterplot по любым двум фичам.")


# Заголовок в стиле
st.markdown(
    """
    <h2 style='text-align: center; color: red;'>🚧 Эта страница находится в разработке! 🚧</h2>
    <p style='text-align: center; color: gray;'>Пожалуйста, возвращайтесь позже.</p>
    """,
    unsafe_allow_html=True
)

# Используем spinner для уведомления о разработке
with st.spinner('Страница загружается...'):
    import time
    time.sleep(2)  # Имитация задержки

st.success("Спасибо за ваше терпение! 😊")
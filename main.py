import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Название
#Описание

st.title("Сервис для рассчета регрессии")
st.write("Здесь вы можете получить результаты линейной или логистической регрессии для тренировочной или тестовой выборки, а также возможность построить scatterplot по любым двум фичам.")

st.write("Детали и более четки инструкции будут чуть позже.")
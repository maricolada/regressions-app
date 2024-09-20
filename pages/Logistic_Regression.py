import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Название
#Описание

st.title("Логистическая регрессия и визуализация данных")
st.write("Здесь вы можете получить результат логистической регрессии для тренировочной или тестовой выборки, а также возможность построить scatterplot по любым двум фичам.")

# Шаг 1. Загрузка файла

uploaded_file = st.file_uploader("Загрузите файл тренировочной или тестовой выборки", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Полученные данные для анализа")
    st.write(data)

    # Шаг 2. Выбор переменных

    ## Выбор зависимой переменной (таргет)
    target_column = st.selectbox("Укажите target (целевая переменная)", options=data.columns)

    ## Выбор признаков для регрессии (фичей)
    features = data.columns.drop(target_column)
    selected_features = st.multiselect("Выберите не более двух features (признаки для регрессии)", options=features)

    if st.button("Рассчитать регрессию"):
        
        # Шаг 3. Подготовка данных и запуск регрессии
        
        if target_column and len(selected_features) <= 2:
             
             ## делим данные на X (независимые признаки, фичи) и y (целевая переменная)
             X = data[selected_features]
             y = data[target_column]

             ## Делим данные на тренировочную и тестовую выборки
             
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

             ## Обучаем модель логистической регрессии

             lr = LogisticRegression()
             lr.fit(X_train, y_train)

             ## Сделаем предсказания на тестовом наборе данных

             y_pred = lr.predict(X_test)

             ## Оценка модели

             accuracy = accuracy_score(y_test, y_pred)
             st.write(f"Точность модели: {accuracy:.2f}")

             ## Отображение отчета о классификации

             st.subheader("Отчет о классификации")
             report = classification_report(y_test, y_pred, output_dict=True)
             st.write(pd.DataFrame(report).transpose())

             ## Получаем коэффициенты и перехват

             coef = lr.coef_[0]
             intercept = lr.intercept_[0]

             ## Создаем словарь весов признаков

             weight_dict = {selected_features[i]: coef[i] for i in range(len(coef))}
             weight_dict['Целевая переменная, когда все признаки для регрессии равны нулю'] = intercept  # Добавление перехвата в словарь

             ## Отображаем словарь весов признаков

             st.subheader("Словарь весов признаков")
             st.write(weight_dict)


             # Шаг 4. Визуализация данных регрессии

             ## Определяем диапазон для оси X
             x_range = np.linspace(X[selected_features[0]].min(), X[selected_features[0]].max(), 200)

             ## Определяем значения Y для линии решения
             if len(selected_features) == 2:

                y_bound = -(intercept + coef[0] * x_range) / coef[1]

                ## График
                plt.figure(figsize=(14, 8))

                ## Точки из тренировочного набора
                scatter = plt.scatter(X_train[selected_features[0]], X_train[selected_features[1]], 
                                      c=y_train,  # Используем целевую переменную для окраски
                                      cmap='coolwarm', 
                                      s=100, alpha=0.7)

                ## Разделяющая прямая
                plt.plot(x_range, y_bound, color='black', linewidth=3, label='Decision Boundary')

                ## Цветовая шкала
                cbar = plt.colorbar(scatter, orientation='horizontal')
                cbar.set_label(f'Personal Loan (вероятность отказа (0) = Blue, вероятность одобрения (1) = Red)', fontsize=12)

                plt.title(f'Логистическая регрессия: вероятность получения {target_column} в зависимости от {selected_features[0]} и {selected_features[1]}', fontsize=16)
                plt.xlabel(selected_features[0], fontsize=14)
                plt.ylabel(selected_features[1], fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.8)
                plt.style.use('ggplot')
                plt.legend()
                st.pyplot(plt)

                # Шаг 5. Скачать график
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                st.download_button(
                    label="Скачать график",
                    data=buf,
                    file_name="logistic_regression_plot.png",
                    mime="image/png"
                    )


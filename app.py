import os
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(current_dir, 'final_model_medinsuranse.joblib')
model = None

if os.path.exists(MODEL_FILENAME):
    try:
        model = load(MODEL_FILENAME)
        st.caption('Модель загружена')
    except Exception as e:
        st.error(f'Ошибка при загрузке модели: {e}')
else:
    st.error(f'Файл модели не найден')

st.set_page_config(page_title='Insurance LR', page_icon='💊', layout='centered')
st.title('Определение стоимости медицинского страхования')
st.write('Введите ваши параметры для предсказания стоимости страховки')

age = st.slider('Возраст', 18, 65, 30)
sex = st.selectbox('Пол', ['Мужской', 'Женский'])
bmi = st.slider('Индекс массы тела (BMI)', 15.0, 50.0, 25.0)
children = st.slider('Количество детей', 0, 5, 0)
smoker = st.selectbox('Курильщик', ['Да', 'Нет'])
region = st.selectbox('Регион', ['Юго-Запад', 'Юго-Восток', 'Северо-Запад', 'Северо-Восток'])

sex = 1 if sex == 'Мужской' else 0
smoker = 1 if smoker == 'Да' else 0
region_map = {'Юго-Запад': 0, 'Юго-Восток': 1, 'Северо-Запад': 2, 'Северо-Восток': 3}
region = region_map[region]

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

if st.button('Предсказать стоимость страховки'):
    if model is None:
        st.error('Модель не загружена')
    else:
        try:
            prediction = model.predict(input_data)
            st.success(f'Предсказанная стоимость страховки: ${prediction[0]:,.2f}')
        except Exception as e:
            st.error(f'Ошибка при выполнении предсказания: {e}')

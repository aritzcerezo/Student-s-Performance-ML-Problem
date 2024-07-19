import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

directorio_actual = os.getcwd()

# Título de la aplicación
st.title("Predicción del GPA mediante Regresión Lineal")
st.image("image.png",  use_column_width=True)

# Explicación del modelo
st.header("Descripción del Modelo")
st.write("""
Esta aplicación utiliza un modelo de regresión lineal para predecir el GPA (Promedio de Calificaciones) de los estudiantes. 
El modelo ha sido entrenado con datos históricos y considera los siguientes parámetros:
- **Horas de Estudio Semanales:** El número de horas que un estudiante dedica al estudio cada semana.
- **Número de Ausencias:** El número de días que el estudiante ha estado ausente.
- **Nivel de Apoyo Parental:** El nivel de apoyo que el estudiante recibe de sus padres, categorizado como 'Ninguno', 'Bajo', 'Moderado', 'Alto' o 'Muy Alto'.

Puedes ajustar estos parámetros en la barra lateral para ver cómo afectan la predicción del GPA.
""")

# Ingreso de datos del usuario
st.sidebar.header("Parámetros de Entrada del Usuario")

def user_input_features():
    StudyTimeWeekly = st.sidebar.slider("Horas de Estudio Semanales", 0, 20, 5)
    Absences = st.sidebar.slider("Número de Ausencias", 0, 30, 5)
    ParentalSupport = st.sidebar.selectbox("Nivel de Apoyo Parental", 
                                           options=["Ninguno", "Bajo", "Moderado", "Alto", "Muy Alto"])
    

    dict={"Ninguno":"None", "Bajo":"Low", "Moderado": "Moderate", "Alto": "High", "Muy Alto": "Very High"}
    data = {
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "ParentalSupport": dict[ParentalSupport],
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

X = pd.read_csv('data\clean_data\clean_features.csv')
y = pd.read_csv('data\clean_data\clean_target_gpa.csv')['GPA']
X['ParentalSupport'] = X['ParentalSupport'].fillna('None')
X['ParentalEducation'] = X['ParentalEducation'].fillna('None')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

# Cargar el modelo guardado
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Realizar predicciones
prediction_scaled = model.predict(input_df)
prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

# Mostrar los resultados
if prediction[0][0]<0:
    prediction[0][0]=0

st.header(f"Predicción del GPA:     {round(prediction[0][0],2)}")

def respuesta(val):
    if val < 2.0:
        return "El GPA predicho es bastante bajo. Es recomendable aumentar las horas de estudio y reducir las ausencias para mejorar el rendimiento académico."
    elif 2.0 <= val < 3.0:
        return "El GPA predicho es regular. Hay espacio para mejorar. Considere aumentar el apoyo parental y mantener una buena asistencia."
    elif 3.0 <= val < 3.5:
        return "El GPA predicho es bueno. Sigue manteniendo buenos hábitos de estudio y asistencia."
    elif 3.5 <= val < 3.7:
        return "El GPA predicho es muy bueno. ¡Sigue así! Continúa con los buenos hábitos de estudio y asistencia."
    else:
        return "¡Excelente! El GPA predicho es sobresaliente. Continúa con el excelente trabajo y asegúrate de mantener estos hábitos positivos."
    
mensaje_respuesta = respuesta(round(prediction[0][0],2))
st.write(mensaje_respuesta)

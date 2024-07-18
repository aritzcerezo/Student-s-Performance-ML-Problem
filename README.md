# Student-s-Performance-ML-Problem / Estudio de Machine Learning sobre Datos de Estudiantes
![]](image.png)
## Descripción del Proyecto

Este proyecto consiste en dos modelos de machine learning que utilizan datos de estudiantes para predecir el rendimiento académico de los mismos. El primer modelo es un regresor lineal que predice el GPA (Grade Point Average) de los estudiantes. El segundo modelo es un clasificador multiclase que predice automáticamente la clase de calificación del estudiante ('A', 'B', 'C', 'D', 'F').

## Descripción de los Datos

El conjunto de datos contiene la siguiente información sobre los estudiantes:

- **StudentID**: Un identificador único asignado a cada estudiante (1001 a 3392).
- **Age**: La edad de los estudiantes, que varía entre 15 y 18 años.
- **Gender**: Género de los estudiantes, donde 0 representa Masculino y 1 representa Femenino.
- **Ethnicity**: La etnia de los estudiantes, codificada como:
  - 0: Caucásico
  - 1: Afroamericano
  - 2: Asiático
  - 3: Otro
- **ParentalEducation**: El nivel de educación de los padres, codificado como:
  - 0: Ninguno
  - 1: Escuela Secundaria
  - 2: Algo de Universidad
  - 3: Título de Grado
  - 4: Superior
- **StudyTimeWeekly**: Tiempo de estudio semanal en horas, que varía de 0 a 20.
- **Absences**: Número de ausencias durante el año escolar, que varía de 0 a 30.
- **Tutoring**: Estado de tutoría, donde 0 indica No y 1 indica Sí.
- **ParentalSupport**: El nivel de apoyo parental, codificado como:
  - 0: Ninguno
  - 1: Bajo
  - 2: Moderado
  - 3: Alto
  - 4: Muy Alto
- **Extracurricular**: Participación en actividades extracurriculares, donde 0 indica No y 1 indica Sí.
- **Sports**: Participación en deportes, donde 0 indica No y 1 indica Sí.
- **Music**: Participación en actividades musicales, donde 0 indica No y 1 indica Sí.
- **Volunteering**: Participación en voluntariado, donde 0 indica No y 1 indica Sí.
- **GPA**: Promedio de calificaciones en una escala de 0.0 a 4.0, influenciado por hábitos de estudio, participación parental y actividades extracurriculares.
- **GradeClass**: Clasificación de las calificaciones del estudiante basada en el GPA:
  - 0: 'A' (GPA >= 3.5)
  - 1: 'B' (3.0 <= GPA < 3.5)
  - 2: 'C' (2.5 <= GPA < 3.0)
  - 3: 'D' (2.0 <= GPA < 2.5)
  - 4: 'F' (GPA < 2.0)

## Modelos de Machine Learning

### Modelo de Regresión Lineal

Este modelo se utiliza para predecir el GPA de los estudiantes basado en las características proporcionadas. La regresión lineal es una técnica que modela la relación entre una variable dependiente y una o más variables independientes.

### Modelo de Clasificación Multiclase

Este modelo clasifica automáticamente a los estudiantes en una de las cinco clases de calificación ('A', 'B', 'C', 'D', 'F') basándose en las características proporcionadas. El modelo de clasificación multiclase se entrenó utilizando algoritmos de clasificación que pueden manejar más de dos clases.

## Resultados

Ambos modelos fueron evaluados utilizando métricas adecuadas para cada tipo de modelo. Los resultados muestran la capacidad de estos modelos para predecir el rendimiento académico de los estudiantes, proporcionando información valiosa para mejorar la educación y el apoyo a los estudiantes.

## Descripción de los archivos

1. **data_cleaning**: Es un Jupyther Notebook en el que se importan los datos y se limpian para la creación y prodesado de los modelos.
2. **gpa_predictor**: Es un Jupyther Notebook en el que se importan los datos limpios y se prueban y analizan distintos Regresores Lineales para la obtención de un modelo consistente que predice el GPA de los alumnas y que ha sido guardado como **linear_regression_model**.
3.**gradeClass_predictor**: Es un Jupyther Notebook en el que se importan los datos limpios y se crea un clasificador multiclase que predice la clase de las calificaciones de los alumnos mediante RandomForest.

## Conclusión

Este proyecto demuestra cómo el machine learning puede ser aplicado para predecir y clasificar el rendimiento académico de los estudiantes basándose en una variedad de factores. Los modelos desarrollados pueden ayudar a identificar estudiantes que necesitan apoyo adicional y a mejorar las estrategias educativas.


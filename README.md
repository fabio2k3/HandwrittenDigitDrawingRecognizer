# Handwritten Digit Drawing Recognizer

## Objetivo del Proyecto

El objetivo de este proyecto es crear una **inteligencia artificial capaz de reconocer dígitos manuscritos** dibujados por el usuario. El sistema utiliza **redes neuronales convolucionales (CNN)** entrenadas con el dataset **MNIST** para identificar correctamente números del 0 al 9.  

Además, el proyecto permite que el usuario **corrija predicciones incorrectas** y que el modelo **aprenda de estas correcciones**, mejorando su precisión con el tiempo.

## Qué le brinda al usuario

- **Interfaz gráfica interactiva** para dibujar dígitos.
- **Predicción en tiempo real** del número dibujado.
- **Visualización de la probabilidad** de cada dígito (histograma de confianza).
- **Opción de corrección**: si la predicción es incorrecta, el usuario puede indicar la clase correcta.
- **Aprendizaje incremental**: el modelo puede actualizarse con las correcciones del usuario para mejorar su desempeño sin necesidad de reentrenar desde cero.

## Tecnologías utilizadas

- Python 3
- PyTorch
- Tkinter (para la interfaz gráfica)
- Pillow (para manejo de imágenes)
- Matplotlib (para mostrar histograma de probabilidades)
- MNIST dataset

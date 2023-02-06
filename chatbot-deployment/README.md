# Implementación de Chatbot con Flask y JavaScript

En este parte implementamos el chatbot con Flask y JavaScript.

Esto da 2 opciones de implementación:
- Despliegue dentro de la aplicación Flask con la plantilla jinja2

- Servir solo la API de predicción de Flask. Los archivos html y javascript utilizados se pueden incluir en cualquier aplicación Frontend (con solo una ligera modificación) y se pueden ejecutar completamente separados de la aplicación Flask.

## Configuración inicial:
Este repositorio actualmente contiene todos los archivos para la implemntacion del chatbot (Backend y frontend)

### En la carperta Principal contiene lo siguientes archivos:
* nltk_utils.py : donde estan todas las funciones para el preprocesamiento(tokenizacion,eliminacion de stopwords, eliminar signos de puntuacion, lematizacion, BagOfWord)
* train.py : dodne se encuentra la preparacion de los datos y el entrenamiento del algoritmo de machine learning
* model.py: donde se encuentra el modleo de redes neuronales simple.
* chat.py : donde se encuentra el chatbot responsivo 
* app.py : donde encuentra la aplicacion y la conexion con flask y js

### En la carpeta  Static
Se encuentra la conexion con js y el estilo del chatbot con css
### En la carpeta templates
Se encuentra el archivo html pero este tiene problemas en la ejecucion
### En la carpeta standalone-frontend se encuentra la interfaz independiente del backend del chatbot 

### Intalar librerias 
Es recomendable usar anaconda para el entrenamiento del cahtbot

Para la instalación de PyTorch, consulte el sitio web oficial .

Instalar dependencias

$ pip install nltk
$ pip install torchvision 

### CORRER ENTRENAR EL MODELO
Ingresa python

$ python
>>> import nltk
>>> nltk.download('punkt')

Modifique `intents.json` con diferentes intentos y respuestas para su Chatbot

$ python train.py

Esto volcará el archivo data.pth. y luego corre
el siguiente comando para probarlo en la consola.

$ python chat.py

Ahora, para la vista en el frontend
```
$ python app.py
#### EJECUTAR
listo una vez echo todo ejecuta el archivo base.html de standalone-fronted


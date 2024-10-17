# Detección de Personas con YOLOv5 en Python

## Introducción

La detección de objetos es un campo fundamental de la Computer Vision, con aplicaciones que van desde la seguridad o auditorias hasta la conducción autónoma de vehículos (RoboTaxi de Tesla es un claro ejemplo). En particular, la detección de personas es crucial para sistemas de vigilancia o análisis de multitudes. En este mini proyecto, implementamos un detector de personas utilizando **YOLOv5**, uno de los algoritmos de detección de objetos *en tiempo real* más avanzados y accesibles del mundo. Éste lo vamos a aplicar a un conjunto de imágenes (datos: [Mall Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)) para contar el número de personas presentes en cada una.

## ¿Qué es YOLOv5?

**YOLOv5** (You Only Look Once version 5) es una versión mejorada de la familia de modelos YOLO para la detección de objetos. YOLO, principalmente se caracteriza por su capacidad para realizar detecciones de objetos con alta precisión. A diferencia de otros métodos que requieren múltiples pasadas sobre una imagen, YOLO predice las coordenadas de los cuadros delimitadores y las clases de los objetos en una sola pasada, lo que lo hace extremadamente eficiente.

## Objetivo del Proyecto

El objetivo de este proyecto es, a parte de aprender a usar YOLO, a como:
- Implementar un detector de personas utilizando el modelo preentrenado YOLOv5.
- Aplicar el detector a un conjunto de imágenes y contar el número de personas en cada imagen.
- Visualizar los resultados mostrando los cuadros delimitadores y el conteo total de personas detectadas.

## Metodología

### Requisitos Previos

A día hoy, Julio de 2024, los requisitos previos quedan tal que así:
- **Python 3.6** o superior.
- **PyTorch**: Biblioteca de Deep Learning que nos proporciona soporte para la implementación y entrenamiento de modelos.
- **OpenCV**: Librería para procesamiento de imágenes y visión por computadora.
- **Matplotlib**: Biblioteca para crear visualizaciones estáticas y dinámicas en Python.

### Instalación de Dependencias

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
```

### Descarga del Modelo YOLOv5

Utilizamos el repositorio oficial de [Ultralytics](https://docs.ultralytics.com/) para cargar el modelo preentrenado:

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

### Explicación Código Completo

A continuación, se presenta el código completo con comentarios que explican cada sección.

```python
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import random
import os
import warnings

# Suprimimos las advertencias de funciones obsoletas (ojo con esto!)
warnings.filterwarnings('ignore', category=FutureWarning)

print('Entorno activado actual:', sys.executable)
print('Torch-version:\t', torch.__version__)
print('OpenCV-version:\t', cv2.__version__)

# Verificar si CUDA está disponible (por si podemos usar GPU en vez de CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsando el dispositivo: {device.upper()}")

# Ruta al archivo zip del conjunto de datos y ruta de extracción (me lo he descargado en un zip)
zip_file_path = 'C:/ruta/al/archivo/mall_dataset.zip'
extract_path = 'C:/ruta/de/extraccion/'

# Extraemos el archivo zip (descomentar en la primera ejecución)
# import zipfile
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# Ruta a las imagenes extraidas
path_frames = os.path.join(extract_path, 'mall_dataset', 'frames')

# Lista de los archivos con imágenes
extracted_files = []
for root, dirs, files in os.walk(path_frames):
    for file in files:
        if file.endswith(".jpg"):
            extracted_files.append(os.path.join(root, file))

print(f"Número de imágenes extraídas: {len(extracted_files)}")
# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Selección de  10 imágenes aleatorias
random_images = random.sample(extracted_files, 10)

# Creamos una función para detectar y mostrar resultados de YOLO
def detect_and_display(image_path):

    # Leemos la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al leer la imagen: {image_path}")
        return

    # Realizamos predicciones
    results = model(img)

    # Obtenemos las detecciones de personas
    detections = results.pandas().xyxy[0]
    people_detections = detections[detections['name'] == 'person']

    # Contamos las personas detectadas
    total_people = len(people_detections)

    # Dibujanos cuadros reojos alrededor de cada persona
    for _, row in people_detections.iterrows():
        x_min = int(row['xmin'])
        y_min = int(row['ymin'])
        x_max = int(row['xmax'])
        y_max = int(row['ymax'])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Añadimos el texto con el conteo total
    text = f'Personas detectadas: {total_people}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 35), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostramos la imagen
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Aplicamos la función a las imágenes seleccionadas
for img_path in random_images:
    detect_and_display(img_path)
```

### Explicación del Código

- **Carga de librerías**: Importamos las librerías necesarias y suprimimos advertencias innecesarias para mantener la salida limpia.
- **Configuración del dispositivo**: Verificamos si hay una GPU disponible para utilizar CUDA y acelerar el procesamiento.
- **Carga del conjunto de datos**: Especificamos la ruta al conjunto de datos (en este caso, el "Mall Dataset") y listamos todas las imágenes disponibles.
- **Carga del modelo**: Utilizamos `torch.hub` para cargar el modelo YOLOv5 preentrenado, lo que nos evita tener que entrenar el modelo desde cero.
- **Función `detect_and_display`**: Esta función realiza las siguientes tareas:
  - **Lectura de la imagen**: Utiliza OpenCV para leer la imagen desde la ruta proporcionada.
  - **Predicciones**: Pasa la imagen al modelo YOLOv5 para obtener las detecciones.
  - **Filtrado de detecciones**: Solo mantenemos las detecciones que corresponden a personas.
  - **Conteo y visualización**: Contamos el número de personas detectadas, dibujamos cuadros delimitadores alrededor de ellas y añadimos un texto con el conteo total.
  - **Visualización**: Utilizamos Matplotlib para mostrar la imagen resultante.
- **Aplicación de la función**: Iteramos sobre una muestra aleatoria de imágenes y aplicamos la función `detect_and_display` a cada una.

## Resultados

Al ejecutar el código, el modelo detecta y cuenta el número de personas en cada una de las imágenes seleccionadas aleatoriamente. A continuación mostramos dos ejemplos de las detecciones realizadas:

![Ejemplo de detección 1](https://github.com/OriolGilabertLopez/MachineLearning/blob/bb4ec3f11bc05992053716901f5d21501467611c/Projects/MachineLearning/ImageDetection/YoloImages/image1.png)

En la primera imagen se muestra el centro comercial del conjuto de datos de *Mall Dataset*, donde se han detectado **16 personas**. Los cuadros rojos alrededor de las personas nos indican las detecciones realizadas por el modelo YOLOv5. A continuación podem sacar las siguientes conclusiones:

- **Detección de personas**: El modelo ha detectado correctamente a varias personas distribuidas en el área visible de la imagen. Esto incluye personas caminando tanto en primer plano como en el fondo.
- **Precisión de los cuadros delimitadores**: Los cuadros rojos abarcan a las personas detectadas en su mayoría con precisión. Algunas personas en el fondo o parcialmente cubiertas han sido detectadas, lo cual es un buen indicador del rendimiento del modelo.


![Ejemplo de detección 1](https://github.com/OriolGilabertLopez/MachineLearning/blob/bb4ec3f11bc05992053716901f5d21501467611c/Projects/MachineLearning/ImageDetection/YoloImages/image2.png))

La conclusiones son las mismas que en la anterior imagen, pero en este caso se han detectado **21 personas**

## Conclusión

Este proyecto se ha visto cómo utilizar YOLOv5 para la detección de personas en imágenes de manera eficiente y con alta precisión. La simplicidad de la implementación, gracias a PyTorch y al repositorio de Ultralytics, nos ha permitido obtener resultados precisos con poco código (muy útil para el mantenimiento en producción).

## Referencias

- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)


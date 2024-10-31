# Detección de Personas con YOLOv5 en Python

## Introducción

La detección de objetos es un campo fundamental de la Computer Vision, con aplicaciones que van desde la seguridad o auditorias hasta la conducción autónoma de vehículos (RoboTaxi de Tesla es un claro ejemplo). En particular, la detección de personas es crucial para sistemas de vigilancia o análisis de multitudes. En este mini proyecto, implementamos un detector de personas utilizando **YOLOv5**, uno de los algoritmos de detección de objetos *en tiempo real* más avanzados y accesibles del mundo. Éste lo vamos a aplicar a un conjunto de imágenes (datos: [Mall Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)) para contar el número de personas presentes en cada una.

## ¿Qué es YOLOv5?

**YOLOv5** (You Only Look Once version 5) es una versión mejorada de la familia de modelos YOLO (más info [aquí](https://github.com/ultralytics/yolov5)) para la detección de objetos. YOLO, principalmente se caracteriza por su capacidad para realizar detecciones de objetos con alta precisión. A diferencia de otros métodos que requieren múltiples pasadas sobre una imagen, YOLO predice las coordenadas de los cuadros delimitadores y las clases de los objetos en una sola pasada, lo que lo hace extremadamente eficiente y fácil de implementar.

## Objetivo del Proyecto

El objetivo de este proyecto es, a parte de aprender a usar YOLO, ver como podemos:
- Implementar un detector de personas utilizando el modelo preentrenado YOLOv5.
- Aplicar un detector a un conjunto de imágenes y contar el número de personas en cada imagen (se pueden contar animales, etc...).
- Visualizar los resultados mostrando los cuadros delimitadores y el conteo total de personas detectadas.

## Metodología

### Requisitos Previos

A día hoy, a Julio de 2024, los requisitos previos quedan tal que así:
- **Python 3.6** o superior.
- **PyTorch**: Biblioteca de Deep Learning que nos proporciona soporte para la implementación y entrenamiento de modelos.
- **OpenCV**: Librería para procesamiento de imágenes y visión por computadora.
- **Matplotlib**: Biblioteca para crear visualizaciones estáticas y dinámicas en Python.

### Instalación de Dependencias

Debemos asegurarnos de tener instaladas las siguientes librerías:

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
```

### Selección y descarga del Modelo YOLOv5

Para poder descargar y utilizar YOLO, nos iremos al repositorio oficial de [Ultralytics](https://docs.ultralytics.com/), donde obtendremos el modelo preentrenado:

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
```

En el repositorio de **YOLOv5** de **Ultralytics**, vemos que se proporcionan diferentes versiones del modelo preentrenado, cada una optimizada para un equilibrio distinto entre **velocidad** y **precisión**. Las versiones disponibles son:

- **yolov5s (small)**: La versión más pequeña y rápida, pero menos precisa.
- **yolov5m (medium)**: Una versión intermedia entre velocidad y precisión.
- **yolov5l (large)**: Más grande y más precisa, pero más lenta.
- **yolov5x (extra large)**: La versión más grande, más precisa, pero también la más lenta.

Para nuestro ejemplo, usaremsos `yolov5m` dado que este nos ofrece un equilibrio óptimo entre tamaño, velocidad y precisión, lo cual es ideal para nosotros. Incluso para soluciones comerciales que requieren detección en tiempo casi real, `yolov5m` es un buen modelo gracias a su buen rendimiento.


### Explicación del código

A continuación, vamos a ver el código completo y lo vamos a explicar paso a paso :). 

#### Entrorno de ejecución:

Primero, debemos comprobar que el entorno de ejecucion del modelo es el correcto


```python
import sys
print('Entorno actual:', sys.executable)
```
Salida:
```
Entorno actual: c:\Users\usuario\AppData\Local\anaconda3\envs\env_YOLO_model\python.exe

```
Nos aseguramos que los **warnings** no salgan (¡cuidado con esto!)

```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```
básicamente suprimimos las advertencias de funciones obsoletas, lo cual para una presentacion visual es correcto, pero ojo si lo que se desea es productivizar el modelo.

#### Comprobación de las versiones de PyTorch y OpenCV:

```python
import torch
import cv2

print('Entorno activado actual:', sys.executable)
print('Torch-version:\t', torch.__version__)
print('OpenCV-version:\t', cv2.__version__)
```
Salida:
```
Torch-version:	 2.5.0
OpenCV-version:	 4.10.0
```

Ahora, verificamos si `CUDA` está disponible (por si podemos usar GPU en vez de CPU)
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsando el dispositivo: {device.upper()}")
```
Salida:
```
Usando el dispositivo: CPU
```

#### Importacion de los datos:

Ahora, definimos la ruta al archivo zip del conjunto de datos de **Mall** y la ruta de extracción donde se guardaran los frames
```python
zip_file_path = 'C:/ruta/al/archivo/mall_dataset.zip'
extract_path = 'C:/ruta/de/extraccion/'
```

y, como nos lo hemos descargado con fomrato .zip, debemos descomprimir el archivo

```python
import zipfile
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

Para ello, creamos una ruta completa hacia el directorio que contiene los frames del conjunto de datos, de esta forma nos aseguramos de que los _separadores de directorio_ sean correctos para el sistema.

```python
import os
path_frames = os.path.join(extract_path, 'mall_dataset', 'frames')
```

Ahora, vamos a recorrer el directorio `path_frames` (y sus subdirectorios) y vamos a almacenar en una lista todos los archivos con extensión `.jpg` 
```python
extracted_files = []
for root, dirs, files in os.walk(path_frames):
    for file in files:
        if file.endswith(".jpg"):
            extracted_files.append(os.path.join(root, file))
```

El outpuy de `extracted_files` seria este:
```
['..../output/mall_dataset/frames/seq_000001.jpg',
 '..../output/mall_dataset/frames/seq_000002.jpg',
 '..../output/mall_dataset/frames/seq_000003.jpg',
 '..../output/mall_dataset/frames/seq_000004.jpg',
 '..../output/mall_dataset/frames/seq_000005.jpg',
...
]
```
Y, de todas estas, solo seleccionamos 10 al azar para ilustrar este ejemplo:

```python
 import random
random_images = random.sample(extracted_files, 10)
```

#### Carga del modelo `YOLOv5m`

```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```
Salida:

```
Using cache found in C:\Users\usuario/.cache\torch\hub\ultralytics_yolov5_master
YOLOv5  2024-10-17 Python-3.12.7 torch-2.5.0 CPU

Fusing layers... 
YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs
Adding AutoShape... 
```

#### Definición de la funcón principal: `detect_and_display(image_path)`:

La función `detect_and_display(image_path)` es una implementación diseñada para procesar los frames y realizar la detección de personas utilizando usando dicho modelo. El objetivo principal es identificar todas las personas de las 10 imagenes , resaltar su presencia con marcadores/delimitadores rojos y mostrar el nivel de confianza del modelo para cada detección.


```python

import matplotlib.pyplot as plt

# Creamos una función para detectar y mostrar resultados de YOLO
def detect_and_display(image_path):
    # Cargamos la imagen desde el archivo
    img = cv2.imread(image_path)

    # Realizamos las predicciones con el modelo YOLOv5
    results = model(img)

    # Filtramos las detecciones para quedarnos solo con las personas
    detections = results.pandas().xyxy[0]
    people_detections = detections[detections['name'] == 'person']

    # Contamos cuántas personas hemos sido capaz de detectaron en la imagen
    total_people = len(people_detections)

    # Dibujamos los marcos rojos alrededor de cada persona detectada y añadimos el porcentaje de confianza
    for i, row in people_detections.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence'] * 100  # Convertimos la confianza a porcentaje
        label = f'{confidence:.1f}%'  # Mostramos solo el porcentaje de confianza (ej. "98.5%")

        # Dibujamos el marco rojo alrededor de la persona
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Configuramos el texto para mostrar la confianza
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Reducimos el tamaño del texto
        font_thickness = 1  # Hacemos el texto más fino
        text_color = (255, 255, 255)  # Color blanco para el texto
        text_bg_color = (0, 0, 255)  # Fondo rojo detrás del texto
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size

        # Dibujamos el fondo rojo para el texto justo por encima del cuadro
        cv2.rectangle(img, (x_min, y_min - text_h - 5), (x_min + text_w, y_min), text_bg_color, -1)

        # Añadimos el texto con la confianza sobre el cuadro
        cv2.putText(img, label, (x_min, y_min - 5), font, font_scale, text_color, font_thickness)

    # Añadimos el numero total de personas detectadas en la parte superior de la imagen (top)
    total_text = f'Personas detectadas: {total_people}'
    total_font = cv2.FONT_HERSHEY_SIMPLEX
    total_font_scale = 0.9  
    total_font_thickness = 2  
    total_text_color = (255, 255, 255)
    total_text_bg_color = (0, 0, 255) 
    total_text_size, _ = cv2.getTextSize(total_text, total_font, total_font_scale, total_font_thickness)
    total_text_w, total_text_h = total_text_size

    # Dibujamos el fondo rojo para el texto del total de personas
    cv2.rectangle(img, (10, 10), (10 + total_text_w, 10 + total_text_h + 10), total_text_bg_color, -1)

    # Ajustamos el texto del total de personas en la imagen
    cv2.putText(img, total_text, (10, 30 + total_text_h // 2), total_font, total_font_scale, total_text_color, total_font_thickness)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Realizamos predicciones para 10 imágenes seleccionadas al azar
for img_path in random_images:
    detect_and_display(img_path)

```

### Explicación del Código de forma más detallada:

- **Cargamos las librerías**: 
  - Importamos las librerías necesarias como `matplotlib` para la visualización de imágenes, `random` para la selección de imágenes aleatorias, `cv2` para la manipulación de imágenes con OpenCV, y `torch` para el modelo YOLOv5.
  
- **Primera configuración**: 
  - Verificamos si el sistema tiene disponible una **GPU con CUDA** para acelerar el procesamiento. Si no hay GPU, se utiliza la CPU. Esto nos permite que el código sea flexible y se ejecute en diferentes configuraciones de hardware, nada mas.

- **Leemos el conjunto de datos**: 
  - Especificamos la ruta donde se encuentran las imágenes del **Mall Dataset**, extraemos las rutas de los archivos JPG de los frames/fotos, y los almacenamos en la lista: `extracted_files`.

- **Cargamos el modelo YOLOv5m**: 
  - Utilizamos `torch.hub` para cargar el modelo **YOLOv5 preentrenado**. Esto nos permite usar un modelo que ya ha sido entrenado en el conjunto de datos COCO para detectar múltiples clases de objetos, incluida la clase "persona". Al usar este modelo preentrenado, no es necesario entrenar el modelo desde cero, lo que ahorra mucho tiempo.

- **Creamos la función `detect_and_display`**: 
  - Esta se encarga de aplicar el modelo y realizar las detecciones y mostrar las imágenes con los marcos junto su confianza (precision). Más en detalle:
  
  - **Lectura de la imagen**: 
    - Usamos **OpenCV (`cv2.imread`)** para leer las imágenes desde las rutas de los archivos.

  - **Predicciones con YOLOv5**: 
    - Una vez leída la imagen, la pasamos al modelo YOLOv5 para hacer las predicciones, el cual nos devolverá múltiples detecciones de objetos, pero filtraremos para quedarnos solo con las detecciones de personas.

  - **Conteo de personas**: 
    - Contamos cuántas personas fueron detectadas y almacenamos este número en `total_people`. Este valor lo muestremos al final de cada imagen en la parte superior.

  - **Marcos delimitadores**: 
    - Para cada persona detectada, dibujamos un **marco rojo** alrededor de su posición en la imagen usando `cv2.rectangle()`. Estos cuadros delimitadores nos permitiran marcan visualmente las detecciones del modelo (esto mola :)).
  
  - **Métrica de confianza (precisión)**:
    - Junto al marco delimitador, añadimos el texto que indicará el **porcentaje de confianza** del modelo en esa detección (es un forma de evaluar qué tan seguro está el modelo de que el objeto detectado sea una persona)

  - **Conteo total de personas detectadas**:
    - En la parte superior izquierda de la imagen, añadimos un texto que muestra el número total de personas detectadas en esa imagen (`total_people`). Este texto también lo vamos a colocar sobre un fondo rojo para asegurarnos de que se vea correctamente.

  - **Visualización**: 
    - Una vez que hayamos dibujado los marcos y el texto, utilizaremos **Matplotlib** para mostrar la imagen resultante con sus marcos etc..

- **Implementación de la función**: 
  - Finalmente, iteramos sobre una muestra aleatoria de 10 imágenes seleccionadas de `extracted_files` y aplicamos la función `detect_and_display` a cada una. Esto nos permite ver las detecciones y el conteo de personas en varias imágenes diferentes de forma rápida.

## Resultados

Al ejecutar el código, el modelo nos va a detectar y contar el número de personas en cada una de las imágenes seleccionadas aleatoriamente. Para poner un ejemplo, esta seria la salida de las dos pirmeras imagenes procesadas:

![Ejemplo de detección 1](https://github.com/OriolGilabertLopez/MachineLearning/blob/e6b500eadde987ed8f18aa15be4cd49e8089cc79/Projects/MachineLearning/ImageDetection/YoloImages/image1.png)

En esta imagen del centro comercial,  se han detectado con éxito **21 personas**. Los cuadros rojos alrededor de cada persona representan las detecciones realizadas por el modelo **YOLOv5**, acompañadas de su correspondiente porcentaje de confianza. 

Lo que podemos ver que es que:
- La mayoría de las detecciones tienen niveles de confianza elevados, con varios cuadros superando el **70%** de confianza, lo que demuestra la eficacia del modelo en identificar personas en diversos escenarios.
- El modelo es capaz de detectar personas a diferentes distancias y en distintas áreas del centro comercial, desde el fondo hasta el primer plano, mostrando un rendimiento consistente.
- Incluso en áreas más concurridas, con YOLOv5 hemos  logrado identificar múltiples personas cercanas sin perder precisión, lo cual es ideal para aplicaciones en ambientes con multitudes.

Este resultado muestra el potencia de YOLOv5 para manejar entornos complejos y variados, manteniendo un alto nivel de precisión en la detección de personas.


![Ejemplo de detección 2](https://github.com/OriolGilabertLopez/MachineLearning/blob/e6b500eadde987ed8f18aa15be4cd49e8089cc79/Projects/MachineLearning/ImageDetection/YoloImages/image2.png)

La conclusiones son las mismas que en la anterior imagen, pero en este caso se han detectado **19 personas**

## Conclusión

Con este ejemplo hemos visto cómo utilizar los modelos de YOLO para la detección de personas en imágenes de manera eficiente y con alta precisión. La simplicidad de la implementación, gracias a PyTorch y al repositorio de Ultralytics, nos ha permitido obtener resultados precisos con poco código (muy útil para el mantenimiento en producción).

## Referencias

- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)


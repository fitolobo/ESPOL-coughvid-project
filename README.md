## Código utilizado para sustentar el trabajo: "Clasificación Automatizada de Toses para Diagnósticos de COVID-19 usando Redes Neuronales Convolucionales" (2025)

Utilizando ideas de diferentes repositorios, se construye una nueva arquitectura que paraleliza la extracción de convoluciones y características del audio (centroides, energía etc) concatenando en capas internas esta información para construir un clasificador binario. El trabajo es una primera etapa que simplemente evalúa la capacidad del modelo de generalizar la clasificación entre toses y distinguir patrones diferenciales para el caso covid. No se añade la clase neutral ni otro tipo de datos como sintomatología, grupo etario etc. Se obtuvieron resultados superiores al 90% de *accuracy* verificando la eficiencia de concatenar información en 1 y 2 dimensiones. 

Se ha disponibilizado una pequeña aplicación para testear el modelo entrenado.
Si necesitas los pesos del modelo enviar un correo a *rodolfolobo@ug.uchile.cl*

### Aplicación
![](/images/app.png)

### Uso y explicaciones

Uso de notebooks: 

1 - ```data_analysis.ipynb```: te permitirá obtener visualizaciones y estadísticas de los datos contenidos en el dataset. 

2 - ```training_a_model.ipynb```: es un ejemplo de como entrenar un modelo con estos datos. 


### Links de referencia para transformaciones y análisis de los datos

- [LINK 1](https://www.kaggle.com/code/nasrulhakim86/covid-19-screening-from-audio-part-1)

- [LINK 2](https://www.kaggle.com/code/nasrulhakim86/covid-19-screening-from-audio-part-2)

### Dataset 

- [LINK](https://www.kaggle.com/code/sidwc121/covid-cough-positive-extraction)


### Ideas para aplicar - estudiar 

Modelo con mejores métricas: 

- [LINK](https://github.com/mrzaizai2k/Coughvid-19-CRNN-attention/blob/main/coughvid-19-crnn-attention.ipynb)

- [LINK](https://github.com/Klangio/covid-19-cough-classification)

- [LINK 3](https://pub.towardsai.net/how-did-binary-cross-entropy-loss-come-into-existence-68e38509d2b)

- [LINK 4](⁠https://gombru.github.io/2018/05/23/cross_entropy_loss/)

### Chromagram 
[LINK Imagen!](https://en.wikipedia.org/wiki/Chroma_feature#/media/File:ChromaFeatureCmajorScaleScoreAudioColor.png)


## Autores 

**Profesores Orientadores**: Rodolfo Anibal Lobo, Enrique López

**Estudiante**: Joel Castro

**Institución**: Escuela Superior Politécnica del Litoral (ESPOL)

**Link**: [LINK TESIS](https://www.dspace.espol.edu.ec/handle/123456789/65681)
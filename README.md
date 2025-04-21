## Código utilizado para sustentar el trabajo: "Clasificación Automatizada de Toses para Diagnósticos de COVID-19 usando Redes Neuronales Convolucionales" (2025)

Inspirado en ideas provenientes de distintos repositorios, se desarrolló una nueva arquitectura que paraleliza la extracción de convoluciones y características del audio (como centroides espectrales, energía, entre otras), concatenando esta información en capas internas para construir un clasificador binario. Este trabajo corresponde a una primera etapa cuyo objetivo principal es evaluar la capacidad del modelo para generalizar en la clasificación de toses y detectar patrones distintivos asociados a casos de COVID. No se incluyó la clase "neutral" ni se consideraron otros tipos de datos como sintomatología, grupo etario u otros factores. Los resultados obtenidos alcanzaron una precisión en torno al 90%, evidenciando la eficacia de concatenar información en una y dos dimensiones. Este trabajo fue reconocido con el primer lugar en el concurso *5MinPitch* de la Escuela Superior Politécnica del Litoral (ESPOL).

La arquitectura del modelo se puede representar mediante el siguiente diagrama

![](/images/arquitectura.png)

Se ha disponibilizado una pequeña aplicación para testear el modelo entrenado. Si necesitas los pesos del modelo enviar un correo a *rodolfolobo@ug.uchile.cl*

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

- [LINK 1](https://github.com/mrzaizai2k/Coughvid-19-CRNN-attention/blob/main/coughvid-19-crnn-attention.ipynb)

- [LINK 2](https://github.com/Klangio/covid-19-cough-classification)

- [LINK 3](https://pub.towardsai.net/how-did-binary-cross-entropy-loss-come-into-existence-68e38509d2b)

### Chromagram 
[LINK](https://en.wikipedia.org/wiki/Chroma_feature#/media/File:ChromaFeatureCmajorScaleScoreAudioColor.png)


## Autores 

**Profesores Orientadores**: Rodolfo Anibal Lobo, Enrique López

**Estudiante**: Joel Castro

**Institución**: Escuela Superior Politécnica del Litoral (ESPOL)

**Link**: [LINK TESIS](https://www.dspace.espol.edu.ec/handle/123456789/65681)

**Link**: [Acreditación de Colaboración](images/Carta%20de%20certificación%20para%20Proyectos%20PhD%20Enrique%20Lopez-signed.pdf)
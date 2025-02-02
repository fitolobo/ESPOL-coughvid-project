# Manejo de imports I/O
import os

# Algebra lineal
import numpy as np

# Tablas
import pandas as pd

# IA
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

# Audio
import librosa

# Visualizaciones 
import seaborn as sns
sns.set_palette(sns.color_palette("GnBu_r"))
import matplotlib.pyplot as plt


class AudioDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.file_paths[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Inicializar arrays para el batch
        X = np.zeros(
            (len(batch_paths), N_MELS, int(SAMPLE_RATE * DURATION / HOP_LENGTH), 1)
        )
        y = np.array(batch_labels)

        for i, file_path in enumerate(batch_paths):
            # Cargar y preprocesar el audio
            X[i] = process_audio_file(file_path)

        return X, y


def figura_1(data_full, save_path):
    # Crear figura
    f, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[4, 4]))

    # Personalizar gráficos
    sns.countplot(data=data_full, x="gender", ax=axs[0])
    axs[0].set_title('Distribución por Género')
    axs[0].set_ylabel('Cantidad')

    sns.countplot(data=data_full, x="status", ax=axs[1])
    axs[1].set_title('Distribución por Status')
    axs[1].set_ylabel('Cantidad')

    # Ajustar layout
    f.tight_layout()

    # Guardar figura
    figure_path = os.path.join(save_path, 'distribucion_dataset.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {figure_path}")

    # Cerrar la figura para liberar memoria
    plt.close(f)


def figura_2(data_full, save_path):
    f, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[4, 4]))
    sns.countplot(data=data_full, x="respiratory_condition", ax=axs[0])
    sns.countplot(data=data_full, x="fever_muscle_pain", ax=axs[1])
    f.tight_layout()
    # Guardar figura
    figure_path = os.path.join(save_path, 'distribucion_dataset.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {figure_path}")

    # Cerrar la figura para liberar memoria
    plt.close(f)


def figura_3(data_full, save_path):
    """
    Crea y guarda un gráfico de distribución de cough_detected.
    
    Args:
        data_full (pd.DataFrame): DataFrame con los datos
        save_path (str): Ruta donde guardar la figura
    """
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Crear el gráfico
    g = sns.displot(data_full, x="cough_detected", bins=10, height=4, aspect=3)
    
    # Personalizar el gráfico
    plt.title('Distribución de Detección de Tos')
    plt.xlabel('Probabilidad de Detección de Tos')
    plt.ylabel('Cantidad')
    
    # Guardar la figura
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Cerrar la figura para liberar memoria
    plt.close(g.fig)

def data_analisis(save_path='../images/'):
    """
    Analiza y visualiza la distribución de género y status en el dataset COUGHVID.
    
    Args:
        save_path (str): Ruta donde se guardará la figura generada
    """
    # Crear el directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Definir ruta del dataset
    ROOT = '../input/coughvid-wav/public_dataset/'
    print("Loading dataset metadata ... ")
    
    # Cargar y preparar datos
    data_raw = pd.read_csv(os.path.join(ROOT, 'metadata_compiled.csv'))
    print(f"Dataframe info loaded with size: {data_raw.shape}")
    
    # Llenar valores nulos
    data_full = data_raw.fillna('unknown')
    
    # Figura 1: distribución de genero y condicion
    figura_1(data_full, save_path)
    
    # Figura 2: condicion respiratoria y fiebre-dolor muscular
    figura_2(data_full, save_path)
    
    # Figura 3: Distribución de probabilidad de detección de tos en el audio
    figura_3(data_full, save_path)
    
    return 
import itertools

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def preproces_for_new_architecture(fn_wav):
    # Load audio
    y, sr = librosa.load(fn_wav, mono=True, duration=5)
    
    # Original features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # Add spectrogram (using mel-spectrogram for better representation)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add full chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    
    feature_row = {
        # Original scalar features
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rolloff': np.mean(rolloff),
        'zero_crossing_rate': np.mean(zcr),
        
        # 2D features
        'mel_spectrogram': mel_spec_db,  # Shape will be (n_mels, time)
        'chromagram': chromagram,        # Shape will be (12, time)
    }
    
    # Add MFCC features
    for i, c in enumerate(mfcc):
        feature_row[f'mfcc{i+1}'] = np.mean(c)
    
    # Optional: You might want to normalize or resize the 2D features to a fixed size
    # For example:
    target_length = 128  # Choose a fixed length
    
    # Resize spectrogram and chromagram to fixed dimensions
    if mel_spec_db.shape[1] > target_length:
        feature_row['mel_spectrogram'] = mel_spec_db[:, :target_length]
        feature_row['chromagram'] = chromagram[:, :target_length]
    else:
        # Pad with zeros if shorter
        mel_pad_width = ((0, 0), (0, target_length - mel_spec_db.shape[1]))
        chroma_pad_width = ((0, 0), (0, target_length - chromagram.shape[1]))
        feature_row['mel_spectrogram'] = np.pad(mel_spec_db, mel_pad_width, mode='constant')
        feature_row['chromagram'] = np.pad(chromagram, chroma_pad_width, mode='constant')
    
    return feature_row

# helper functions
def preproces(fn_wav):
    y, sr = librosa.load(fn_wav, mono=True, duration=5) #lee el audio
    
    # calcula diferentes caracteristicas
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)  #calcula una imagen --> notas "musicales"
    rmse = librosa.feature.rms(y=y) #energia
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) #centroide espectral
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr) #ancho de banda
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y) # cuantas veces cruza el cero una señal (ritmo)
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # feature compleja --> información frecuencial relacionada 
                                            # a la escala mel (escala psicoacustica de notas)
                                            # como los seres humanos percibimos diferencias entre dos notas
    
    feature_row = {        
        'chroma_stft': np.mean(chroma_stft), # promedio el chrograma
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rolloff': np.mean(rolloff),
        'zero_crossing_rate': np.mean(zcr),        
    }
    for i, c in enumerate(mfcc):
        feature_row[f'mfcc{i+1}'] = np.mean(c)

    return feature_row


class CoughNet(torch.nn.Module):
    def __init__(self, input_size):
        super(CoughNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.l6 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = self.l6(x)
        return x

# https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(targets, predictions, classes):
    # calculate normalized confusion matrix
    cm = confusion_matrix(targets, predictions)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
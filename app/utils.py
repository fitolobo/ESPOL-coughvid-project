import numpy as np
import librosa


def preproces_for_new_architecture(fn_wav):
    """
        Preprocesamiento de audio. Admite: WAV, OGG, MP3, M4A.
    Args:
        fn_wav (_type_): audio signal

    Returns:
        _type_: audio features
    """
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
    
    # Optional: You might want to normalize or resize the 2D features 
    # to a fixed size
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
        feature_row['mel_spectrogram'] = np.pad(mel_spec_db,
                                                mel_pad_width,
                                                mode='constant')
        feature_row['chromagram'] = np.pad(chromagram,
                                           chroma_pad_width,
                                           mode='constant')
    return feature_row

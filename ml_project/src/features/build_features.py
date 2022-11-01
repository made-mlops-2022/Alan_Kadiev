import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict

# Wav Features and Visualization
import librosa
# Basics
import numpy as np
# Visualization
import matplotlib.pyplot as plt
plt.style.use('ggplot')

train_path = Path('./train/train')
test_path = Path('./test/test')
SEED = 42
test_size = 0.3


params = {
'sample_rate': 22050,
'hop_length': 220,
'n_fft': 3048,
'n_frames_per_example': 1
}

def calc_means(spec):
  res = []
  res = []

  for arr in spec:
    res.append(np.array(arr).mean())
  
  return np.array(res).mean()

def load_wav(path: str, sample_rate: int) -> np.ndarray:
    waveform, y = librosa.load(path, sr=sample_rate)
    return waveform

def extract_features(
        file_path: str, 
        sample_rate: int=22050,
        hop_length: int=220,
        n_fft: int=2048,
        n_frames_per_example: int=1
    ) -> List[float]:
    
    waveform = load_wav(file_path, sample_rate=sample_rate)
    
    spectrogram = librosa.feature.melspectrogram(
        waveform, n_fft=n_fft, hop_length=hop_length
    )

    spectrogram = np.log(1e-20 + np.abs(spectrogram ** 2))

    spectrogram2 = librosa.feature.rms(
        waveform, hop_length=hop_length
    )

    spectrogram2 = np.log(1e-20 + np.abs(spectrogram2 ** 2))

    spectrogram += spectrogram2

    spectrogram3 = librosa.feature.spectral_centroid(
       waveform, n_fft=n_fft, hop_length=hop_length
    )

    spectrogram3 = np.log(1e-20 + np.abs(spectrogram3 ** 2))
    
    spectrogram += spectrogram3

    spectrogram4 = librosa.feature.spectral_bandwidth(
       waveform, n_fft=n_fft, hop_length=hop_length
    )

    spectrogram4 = np.log(1e-20 + np.abs(spectrogram4 ** 2))

    spectrogram += spectrogram4

    n_examples = spectrogram.shape[1] // n_frames_per_example

    
    return [
        spectrogram[
            :,
            i*n_frames_per_example:(i+1) * n_frames_per_example
        ].reshape(1, -1)
        for i in range(n_examples)
    ]

def load_folder_data(
        path: Path, 
        train: bool, 
        params: Dict[str, int]):
    
    features: List[List[np.ndarray]] = []
    labels: List[str] = []
    file_names: List[str] = list(path.rglob('*.wav'))

    for file_path in file_names:
        
        file_features = extract_features(file_path, **params)

        features.append(file_features)
        
        if train:
            class_name = file_path.parent.name
            labels.append([class_name] * len(file_features))
    
    return features, labels, file_names
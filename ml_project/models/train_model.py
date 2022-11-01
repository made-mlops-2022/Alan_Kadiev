from pathlib import Path
from typing import List, Tuple, Dict

# Wav Features and Visualization
import librosa
# Basics
import numpy as np
# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

params = {
    'sample_rate': 22050,
    'hop_length': 220,
    'n_fft': 3048,
    'n_frames_per_example': 1
}

train_path = Path('./train/train')
test_path = Path('./test/test')
SEED = 42
test_size = 0.3

class_names = [folder.name for folder in train_path.iterdir()]

class_name2id = {
    class_name: class_id 
    for class_id, class_name in enumerate(class_names)
}

features, labels, file_names = load_folder_data(train_path, train=True, params=params)

labels = [
    [class_name2id[label] for label in label_list]
    for label_list in labels
]

features_train, features_test, labels_train, labels_test, files_train, files_test = \
train_test_split(
    features, labels, file_names, test_size=test_size, random_state=SEED
)

X_train = np.vstack([x for list_ in features_train for x in list_])
y_train = np.array([x for list_ in labels_train for x in list_])

features_train, features_test, labels_train, labels_test, files_train, files_test = \
train_test_split(
    features, labels, file_names, test_size=test_size, random_state=SEED
)

X_train = np.vstack([x for list_ in features_train for x in list_])
y_train = np.array([x for list_ in labels_train for x in list_])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(solver='sgd'))
]).fit(X_train, y_train)
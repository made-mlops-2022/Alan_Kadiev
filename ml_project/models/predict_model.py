from pathlib import Path

# Basics
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

X_test = np.vstack([
    feature for feature_list in features_test 
    for feature in feature_list
])

y_test = np.array([label for label_list in labels_test for label in label_list])

y_pred = pipe.predict(X_test)


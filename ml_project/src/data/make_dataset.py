# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict

# Wav Features and Visualization
import librosa
#import IPython.display as ipd
# Basics
import numpy as np
import pandas as pd
# Visualization
#import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Machine Learning
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def calc_means(spec):
  res = []
  res = []

  for arr in spec:
    res.append(np.array(arr).mean())
  
  return np.array(res).mean()

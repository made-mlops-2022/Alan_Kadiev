# Homework-1 (TechnoPark)

## Installation

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

## Usage 

Models - RandomForestClassifier or GradientBoostingClassifier:

`export PYTHONPATH=.`

`python ml_project/train_pipeline.py —config-name="train_RFC_config"`

`python ml_project/train_pipeline.py —config-name="train_GBC_config"`

Prediction:

`python ml_project/predict_pipeline.py —config-name="predict_config"`

Tests:

`pytest ml_project/tests/`

## Project Organization

    ├── README.md             <- The top-level README for developers using this project.
    ├── data
    │   └── raw               <- The original, immutable data dump.
    │
    ├── models                <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks             <- Jupyter notebooks with exploratory data analys
    │    
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .)
    │
    └── ml_project                  <- Source code for use in this project
        ├── __init__.py       <- Makes src a Python module
        │
        ├── data              <- Scripts to download or read data
        │
        ├── features          <- Scripts to turn raw data into features for modeling
        │
        ├── models            <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        ├── tests             <- Scripts to test code 
        │
        ├──  train_pipeline.py   <- Scripts to train pipeline
        │
        ├──  predict_pipeline.py <- Scripts to predict pipeline
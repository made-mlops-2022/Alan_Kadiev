from ml_project.data.make_dataset import *
from ml_project.enities import SplittingParams


def test_split_dataset(df: pd.DataFrame):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=42, val_size=val_size,)
    train, val = split_train_val_data(df, splitting_params)
    assert train.shape[0] == 40
    assert val.shape[0] == 10
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# --- compatibility shim: expose train_model for callers that expect that name ---
# If your file already defines a function called `train` (or `run` / `main`), this
# wrapper will call it. If not, it will do nothing and leave the file unchanged.
if 'train_model' not in globals():
    if 'train' in globals():
        def train_model(*args, **kwargs):
            return train(*args, **kwargs)
    elif 'run' in globals():
        def train_model(*args, **kwargs):
            return run(*args, **kwargs)
    elif 'main' in globals():
        def train_model(*args, **kwargs):
            return main(*args, **kwargs)
# --- end shim ---

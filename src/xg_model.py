import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from .features import build_features, XG_FEATURES
from .config import BASE_DIR, RANDOM_STATE

MODEL_PATH = BASE_DIR / "models" / "xg_model.pkl"

def train_xg_model(df: pd.DataFrame, target_col: str = "goals"):
    X, y = build_features(df, XG_FEATURES, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = PoissonRegressor(alpha=0.1, max_iter=1000)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    score = model.score(X_test, y_test)
    return model, score

def load_xg_model():
    return joblib.load(MODEL_PATH)

def predict_team_xg(model, team_features: pd.Series) -> float:
    x = team_features[XG_FEATURES].values.reshape(1, -1)
    return float(model.predict(x)[0])

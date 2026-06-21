import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .features import build_features, OU_FEATURES
from .config import BASE_DIR, RANDOM_STATE

MODEL_PATH = BASE_DIR / "models" / "ou_model.pkl"

def train_ou_model(df: pd.DataFrame, target_col: str = "goals_total"):
    # Luo total-goals jos ei ole
    if target_col not in df.columns:
        df = df.copy()
        df[target_col] = df["goals"] + df["goals_conceded"]

    X, y = build_features(df, OU_FEATURES, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    score = model.score(X_test, y_test)
    return model, score

def load_ou_model():
    return joblib.load(MODEL_PATH)

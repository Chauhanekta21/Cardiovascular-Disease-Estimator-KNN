from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "cardio_train.csv"
MODEL_DIR = APP_DIR / "models"


def main():
    data = pd.read_csv(DATA_PATH, sep=";")
    data = data.rename(
        columns={
            "ap_hi": "systolic_bp",
            "ap_lo": "diastolic_bp",
            "cholesterol": "cholesterol_level",
            "gluc": "glucose_level",
            "smoke": "smoking",
            "alco": "alcohol",
            "active": "physical_activity",
            "cardio": "cardiovascular_disease",
        }
    )
    data["age"] = (data["age"] / 365.25).round().astype(int)
    data = data[(data["height"] >= 100) & (data["height"] <= 220)]
    data = data[data["weight"] >= 30]
    data = data[(data["systolic_bp"] > 0) & (data["systolic_bp"] <= 250)]
    data = data[(data["diastolic_bp"] > 0) & (data["diastolic_bp"] < 250)]
    data = data[data["systolic_bp"] > data["diastolic_bp"]]

    features = data.drop(columns=["id", "gender", "cardiovascular_disease"])
    target = data["cardiovascular_disease"]
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = KNeighborsClassifier(n_neighbors=29, weights="uniform")
    model.fit(x_train_scaled, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "knn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "minmax_scaler.joblib")
    print(f"Saved model and scaler to {MODEL_DIR}")
    print(f"Training rows: {len(x_train)}; features: {list(features.columns)}")


if __name__ == "__main__":
    main()
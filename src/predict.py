from pathlib import Path
import pandas as pd
import json
from src.utils import MODELS_DIR, load_pickle

MODEL_PATH = MODELS_DIR / "model_v1.pkl"
RUL_MODEL_PATH = MODELS_DIR / "rul_model.pkl"
THRESHOLD_PATH = Path("logs/threshold.json")


#simulate temporal features for single-row input
def add_temporal_features_inference(X: pd.DataFrame) -> pd.DataFrame:
    key_sensors = ['sensor_1', 'sensor_5', 'sensor_10']

    for sensor in key_sensors:
        if sensor in X.columns:
            X[f'{sensor}_lag1'] = 0
            X[f'{sensor}_delta'] = 0
            X[f'{sensor}_rolling_mean_5'] = X[sensor]
            X[f'{sensor}_rolling_std_5'] = 0

    return X


class InferencePipeline:

    def __init__(self):
        self.model = load_pickle(MODEL_PATH)
        self.rul_model = load_pickle(RUL_MODEL_PATH)

        # Load trained threshold
        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def _prepare_input(self, data):
        if isinstance(data, dict):
            X = pd.DataFrame([data])
        else:
            X = data.copy()

        # Drop training-only columns
        drop_cols = [col for col in ["label", "RUL"] if col in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        #recreate temporal features
        X = add_temporal_features_inference(X)

        return X

    def predict(self, data):
        X = self._prepare_input(data)

        prob = self.model.predict_proba(X)[0, 1]
        prob = float(prob)

        pred = int(prob >= self.threshold)

        result = {
            "prediction": pred,
            "prediction_label": "near_failure" if pred == 1 else "healthy",
            "failure_probability": max(round(prob, 4), 0.0001),
        }

        result["confidence"] = (
            "high" if prob > (self.threshold + 0.2) else
            "medium" if prob >= self.threshold else
            "low"
        )

        return result

    def predict_rul(self, data):
        X = self._prepare_input(data)

        predicted_rul = float(self.rul_model.predict(X)[0])

        result = {
            "predicted_rul": round(predicted_rul, 2),
            "failure_prediction": "near_failure" if predicted_rul <= 30 else "healthy",
        }

        return result


if __name__ == "__main__":
    pipeline = InferencePipeline()

    test_path = Path("data/processed/test.csv")
    df = pd.read_csv(test_path)

    samples = pd.concat([
        df[df["label"] == 0].sample(5, random_state=42),
        df[df["label"] == 1].sample(5, random_state=42)
    ])

    print("\n--- Sample Predictions ---\n")

    for i, row in samples.iterrows():
        row_df = row.to_frame().T

        cls_result = pipeline.predict(row_df)
        rul_result = pipeline.predict_rul(row_df)

        print(f"Sample {i}:")
        print(f"  Classification: {cls_result}")
        print(f"  RUL Prediction: {rul_result}")
        print()
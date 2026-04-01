from src.data_loader import load_processed_splits, split_features_target
from src.utils import load_pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
from typing import Any

MODEL_PATH = "models/model_v1.pkl"


def evaluate_classifier(model: Any, X, y) -> dict[str, Any]:
    y_pred = model.predict(X)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0),
    }


def save_evaluation_report(all_results: dict[str, Any], report_path: str | Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        for model_name, split_results in all_results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 50 + "\n")
            for split_name, metrics in split_results.items():
                f.write(f"{split_name.capitalize()} metrics:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1: {metrics['f1']:.4f}\n")
                f.write(f"  Confusion matrix: {metrics['confusion_matrix']}\n")
                f.write(f"  Classification report:\n{metrics['classification_report']}\n")
                f.write("\n")
            f.write("\n")


def run_evaluation():
    print("Loading model...")
    model = load_pickle(MODEL_PATH)

    print("Loading data...")
    _, _, test_df = load_processed_splits()

    X_test, y_test = split_features_target(test_df)

    print("Evaluating...")
    metrics = evaluate_classifier(model, X_test, y_test)

    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(metrics["classification_report"])


if __name__ == "__main__":
    run_evaluation()
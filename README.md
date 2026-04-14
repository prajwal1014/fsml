# Industrial Equipment Failure Prediction for Predictive Maintenance

This project implements a machine learning pipeline for predicting equipment failures using the NASA CMAPSS FD001 dataset. It includes data preprocessing, feature engineering, model training, evaluation, and deployment via a FastAPI web service.

## Dataset

**NASA CMAPSS FD001 dataset**  
Link: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation

The dataset contains sensor readings from turbofan engines. Download the FD001 dataset and place it in:

```
data/raw/train_FD001.txt
```

## Project Structure

```
fsml_project_08/
│
├── .github/
│   └── workflows/                 # CI/CD pipeline (GitHub Actions)
│
├── app/
│   ├── app.py                    # FastAPI application (API endpoints)
│   └── schema.py                 # Input validation (Pydantic models)
│
├── data/
│   ├── raw/
│   │   └── train_FD001.txt       # Raw NASA CMAPSS dataset
│   │
│   └── processed/
│       ├── train.csv             # Processed training data
│       ├── val.csv               # Validation data
│       └── test.csv              # Test data
│
├── logs/
│   ├── app.log                   # Application logs
│   ├── evaluation_report.txt     # Model performance summary
│   ├── feature_engineering_notes.json  # Feature explanations
│   ├── model_metrics.json        # Detailed metrics
│   └── threshold.json            # Saved optimal threshold
│
├── models/
│   ├── model_v1.pkl              # Classification model (failure prediction)
│   └── rul_model.pkl             # Regression model (RUL prediction)
│
├── pipeline/
│   └── pipeline.py               # End-to-end pipeline (run everything)
│
├── src/
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocess.py             # Data preprocessing + feature engineering
│   ├── features.py               # Feature engineering logic
│   ├── train.py                  # Model training + selection + threshold tuning
│   ├── evaluate.py               # Model evaluation
│   ├── predict.py                # Inference logic
│   └── utils.py                  # Utility functions
│
├── .dockerignore                 # Ignore files for Docker
├── .gitignore                    # Ignore files for Git
├── Dockerfile                    # Container setup
├── requirements.txt             # Python dependencies
└── README.md                     # Project documentation

```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saiakshay2002/fsml_project_08.git
   cd fsml_project_08
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Dataset Handling
   ```bash
   The dataset is NOT stored in the repository to keep it lightweight.

   Instead, the dataset is automatically downloaded during pipeline execution from Google Drive.

   - Source: NASA CMAPSS FD001 dataset
   - Download is handled inside `pipeline/pipeline.py` using `gdown`

   When you run the pipeline, the dataset will be downloaded automatically to:

   data/raw/train_FD001.txt
   ```

## Usage

### Step-by-Step Workflow

#### 1. Data Preprocessing

Run the preprocessing pipeline to clean and prepare the data:

```bash
python -m src.preprocess
```

This will:
- Load raw data
- Compute Remaining Useful Life (RUL)
- Add binary failure labels (threshold = 30 cycles)
- Split data by engine to prevent data leakage
- Remove low-variance features
- Save processed data to `data/processed/`

#### 2. Model Training

Train multiple models and select the best one:

```bash
python -m src.train
```

This will:
- Load processed data
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate on validation set
- Select best model based on F1-score
- Save best model to `models/model_v1.pkl`
- Save metrics to `logs/model_metrics.json`
- Save evaluation report to `logs/evaluation_report.txt`

#### 3. Model Evaluation

Check the evaluation results:

```bash
cat logs/evaluation_report.txt
```

#### 4. Run Full Pipeline

Alternatively, run the entire pipeline in one command:

```bash
python -m pipeline.pipeline
```

### API Deployment

#### Start the FastAPI server:

```bash
uvicorn app.app:app --host 127.0.0.1 --port 8000 --reload
```

#### Test the API:

- Open Swagger UI: http://127.0.0.1:8000/docs
- Or use curl:

```bash
curl -X POST "http://127.0.0.1:8000/docs" \
     -H "Content-Type: application/json" \
     -d '{ "op_setting_1": 0.003,
  "op_setting_2": 0.0,
  "op_setting_3": 100.0,

  "sensor_1": 520.5,
  "sensor_2": 645.0,
  "sensor_3": 1605.0,
  "sensor_4": 1425.0,
  "sensor_5": 15.2,
  "sensor_6": 22.0,
  "sensor_7": 560.0,
  "sensor_8": 2395.0,
  "sensor_9": 9100.0,
  "sensor_10": 1.5,
  "sensor_11": 55.0,
  "sensor_12": 500.0,
  "sensor_13": 2395.0,
  "sensor_14": 8200.0,
  "sensor_15": 9.2,
  "sensor_16": 0.04,
  "sensor_17": 400.0,
  "sensor_18": 2395.0,
  "sensor_19": 100.0,
  "sensor_20": 42.0,
  "sensor_21": 22.0
}'
```

Sample  response:
```json
{
  "prediction": 1,
  "prediction_label": "near_failure",
  "failure_probability": 0.7889,
  "confidence": "high",
  "predicted_rul": 11.72,
  "rul_based_prediction": "near_failure"
}
```
## Requirements

- Python 3.8+
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.4.2
- xgboost==3.2.0
- fastapi (for API)
- uvicorn (for serving API)

## Logging

All operations are logged to `logs/app.log`. Model training and API requests are logged with timestamps.

## Docker Deployment

### Build Docker Image

```bash
docker build -t ml-model-app .

docker run -p 8000:8000 ml-model-app

```
---


## CI/CD Pipeline (GitHub Actions)

Workflow file:

.github/workflows/main.yml

### Pipeline Steps

On every push to the main branch:

- Install dependencies
- Run ML pipeline (pipeline/pipeline.py)
- Download dataset automatically
- Train model
- Build Docker image

## End-to-End Pipeline

Run full pipeline:

```bash
python pipeline/pipeline.py
```


---

## 📦 Model Versioning

```markdown
## Model Versioning

Model is saved as:

models/model_v1.pkl
```

## Logging

Logs are stored in:

logs/app.log

### Includes

- Pipeline execution logs
- Training logs
- API request logs

## Project Highlights

- End-to-end ML pipeline
- FastAPI deployment
- Docker containerization
- CI/CD using GitHub Actions
- Dynamic dataset download
- Model versioning
- Logging system
- Error logs

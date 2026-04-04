# Industrial Equipment Failure Prediction for Predictive Maintenance

This project implements a machine learning pipeline for predicting equipment failures using the NASA CMAPSS FD001 dataset. It includes data preprocessing, feature engineering, model training, evaluation, and deployment via a FastAPI web service.

## Dataset

**NASA CMAPSS FD001 dataset**  
Link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

The dataset contains sensor readings from turbofan engines. Download the FD001 dataset and place it in:

```
data/raw/train_FD001.txt
```

## Project Structure

```
fsml_project/
│── app/
│   ├── app.py              # FastAPI application
│   └── schema.py           # Pydantic models for API
│
├── data/
│   ├── raw/
│   │   └── train_FD001.txt # Raw dataset
│   └── processed/
│       ├── train.csv       # Processed training data
│       ├── val.csv         # Processed validation data
│       └── test.csv        # Processed test data
│
├── logs/
│   ├── app.log             # Application logs
│   ├── evaluation_report.txt # Model evaluation results
│   ├── feature_engineering_notes.json # Feature documentation
│   └── model_metrics.json  # Detailed metrics
│
├── models/
│   └── model_v1.pkl        # Trained model
│
├── pipeline/
│   └── pipeline.py         # End-to-end pipeline
│
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocess.py       # Data preprocessing
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   ├── predict.py          # Inference pipeline
│   └── utils.py            # Utility functions
│
├── config.yaml             # Configuration (currently empty)
├── Dockerfile              # Docker configuration (currently empty)
├── requirements.txt        # Python dependencies
└── README.md
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

4. Download the dataset and place it in `data/raw/train_FD001.txt`

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
python src/train.py
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
     -d '{
       "sensor_2": 642.0,
       "sensor_3": 1589.0,
       "sensor_4": 1400.0,
       "sensor_7": 554.0,
       "sensor_8": 2388.0,
       "sensor_9": 9056.0,
       "sensor_11": 47.0,
       "sensor_12": 521.0,
       "sensor_13": 2388.0,
       "sensor_14": 8138.0,
       "sensor_15": 8.0,
       "sensor_17": 392.0,
       "sensor_20": 39.0,
       "sensor_21": 23.0
     }'
```

Sample  response:
```json
{
  "prediction": 0,
  "prediction_label": "healthy",
  "failure_probability": 0.1234,
  "confidence": "low"
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
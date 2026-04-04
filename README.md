# Industrial Equipment Failure Prediction for Predictive Maintenance

## Dataset
NASA CMAPSS FD001 dataset  
Link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Download the FD001 dataset and place it in:

data/raw/train_FD001.txt

---

## Project Structure

```text
fsml_project/
│── app/
│   ├── app.py
│   └── schema.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│   └── app.log
│
├── models/
│   └── model_v1.pkl
│
├── pipeline/
│   └── pipeline.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── notebooks/
├── requirements.txt
├── Dockerfile
├── config.yaml
└── README.md
```
---

## Preprocessing
- RUL computation
- Binary failure labeling (threshold = 30)
- Low-variance feature removal
- Engine-wise data split (no leakage)

---

## How to Run Preprocessing

```python
from src.preprocess import preprocess_pipeline

train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")
```
---

## Model Training

### Models Used
- Logistic Regression
- Random Forest
- XGBoost

```python
cd src

python train.py
```
#### Best model saved at models/model_v1.pkl

---

### API Deployment (FastAPI)

#### run server:
```text
uvicorn app.app:app --host 127.0.0.1 --port 8000 --reload
```
#### test:
```text
http://127.0.0.1:8000/docs (Swagger UI)
```
---

#### Full Pipeline

Run everything in one command:
```python
python -m pipeline.pipeline
```
#### Pipeline Flow :- Raw Data → Preprocessing → Feature Engineering → Training → Evaluation → Model Saving
---

#### Outputs
- Processed data → data/processed/
- Model → models/model_v1.pkl
- Logs → logs/app.log


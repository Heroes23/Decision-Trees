# Decision Tree Prediction

A Decision Tree classifier that predicts breast cancer diagnosis (Benign vs. Malignant) from clinical features. Built as part of a supervised machine learning study using a 213-record breast cancer dataset.

---

## End-to-End Workflow

```mermaid
flowchart TD
    A["Breast Cancer Dataset
    S3 CSV · 213 records · 11 columns"]
    B["Load Data
    pd.read_csv"]
    C["Exploratory Data Analysis
    ydata_profiling ProfileReport"]
    D["Lowercase Column Names"]
    E["Age Discretization
    IQR-based → 3 ordinal groups"]
    F["One-Hot Encoding
    all features except target"]
    G["Drop '#' Artifact Columns
    7 columns removed → 247 clean columns"]
    H["Label Encode Target
    Benign = 0 · Malignant = 1"]
    I["Stratified Train / Test Split
    85% train · 15% test"]
    J["Decision Tree Classifier
    gini · max_depth=5 · min_samples_split=3"]
    K["Model Evaluation
    Accuracy · Precision · Recall · F1 · Confusion Matrix"]
    L["Tree Visualization
    graphviz export → PNG"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L
```

---

## Predictive Modeling Strategy

```mermaid
flowchart LR
    subgraph Preprocessing
        direction TB
        P1["Raw Features
        11 columns · 213 rows"]
        P2["Lowercase columns"]
        P3["Age Binning
        min–Q1 → 1
        Q1–Q3  → 2
        Q3–max → 3"]
        P4["One-Hot Encoding
        pd.get_dummies
        246 boolean features"]
        P5["Remove '#' columns
        s/n, year, tumor size, etc."]
        P6["Label Encode Target
        sklearn LabelEncoder"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph Splitting
        direction TB
        S1["Feature matrix X
        246-dim NumPy array"]
        S2["Target vector y
        reshaped -1,1"]
        S3["X_train · y_train
        181 samples · stratified"]
        S4["X_test · y_test
        32 samples · stratified"]
        S1 & S2 --> S3 & S4
    end

    subgraph Model
        direction TB
        M1["DecisionTreeClassifier
        criterion = gini
        splitter  = best
        max_depth = 5
        min_samples_split = 3"]
        M2["dt.fit(X_train, y_train)"]
        M3["y_pred = dt.predict(X_test)"]
        M1 --> M2 --> M3
    end

    subgraph Evaluation
        direction TB
        E1["Confusion Matrix
        TP=18  FP=0
        FN=3   TN=11"]
        E2["Test Accuracy  91%"]
        E3["Precision      100%"]
        E4["Recall         78.6%"]
        E5["F1 Score       0.88"]
        E1 --> E2 & E3 & E4 & E5
    end

    Preprocessing --> Splitting --> Model --> Evaluation
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Age encoding | Ordinal bins (IQR) | Preserves natural ordering; reduces cardinality before OHE |
| OHE strategy | `drop_first=False` | Retains all categories for interpretability in the tree |
| Splitting criterion | Gini impurity | Standard for classification; computationally efficient |
| Test size | 15% stratified | Small dataset (213 rows); stratification preserves class ratio (56% Benign / 44% Malignant) |
| max_depth | 5 | Limits overfitting on a shallow dataset |

---

## Deployment Roadmap

```mermaid
flowchart TD
    A["Trained Model Artifact
    joblib / pickle serialization"]
    B["Model Registry
    MLflow · Weights & Biases · or S3"]

    subgraph Serving
        direction LR
        C1["REST API
        FastAPI endpoint
        POST /predict"]
        C2["Batch Inference
        scheduled pandas pipeline"]
    end

    D["Docker Image
    python:3.11-slim + dependencies"]
    E["Cloud Deployment
    AWS ECS · GCP Cloud Run · Azure ACI"]
    F["Monitoring
    evidently · Grafana"]

    subgraph Feedback_Loop
        direction TB
        G{"Data or performance
        drift detected?"}
        H["Collect new labelled samples
        retrigger preprocessing pipeline"]
        I["Retrain & promote
        CI/CD via GitHub Actions"]
        G -- Yes --> H --> I --> A
        G -- No  --> F
    end

    A --> B --> Serving
    Serving --> D --> E --> F --> Feedback_Loop
```

### Deployment Milestones

- [ ] Serialize trained model with `joblib`
- [ ] Wrap prediction logic in a FastAPI `/predict` endpoint
- [ ] Containerize with Docker; validate locally
- [ ] Push image to ECR / Artifact Registry and deploy
- [ ] Attach evidently dashboard for feature drift and accuracy tracking
- [ ] Wire retraining trigger into GitHub Actions on drift alert

---

## Project Structure

```
zaed_ml_project/
├── dt_model.ipynb              # Exploratory notebook with full output
├── preprocessing.py            # Refactored pipeline script
├── tree_visual_homework_3.png  # Rendered decision tree
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package and environment manager

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and activate the virtual environment

```bash
# Create a .venv in the project root
uv venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `torch`. If you only need the scikit-learn pipeline and want a lighter install, you can skip it:
>
> ```bash
> uv pip install pandas numpy scikit-learn ydata-profiling ipykernel ipywidgets graphviz
> ```

### Run the notebook

```bash
jupyter notebook dt_model.ipynb
```

### Run the script

```bash
python preprocessing.py
```

---

## Results Summary

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Test  (32 samples) | 91.0% | 1.00 | 0.786 | 0.88 |
| Train (181 samples) | 96.7% | 1.00 | 0.924 | 0.961 |

The zero false positives on the test set (Precision = 1.0) mean every predicted Malignant case was correct. The gap between train and test recall (0.924 vs 0.786) indicates mild overfitting — addressable with cross-validation or pruning.

---

## Dataset

| Field | Detail |
|---|---|
| Source | Private S3 bucket (CSV) |
| Records | 213 |
| Features | Age, Menopause, Tumor Size, Inv-Nodes, Breast, Metastasis, Breast Quadrant, History |
| Target | Diagnosis Result — Benign / Malignant |
| Class balance | 120 Benign (56%) · 93 Malignant (44%) |

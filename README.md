# Heart Disease Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare supervised machine learning models that predict the presence of cardiovascular disease in a patient based on clinical attributes. This is a binary classification problem with target variable `target`:

- `1`: presence of heart disease.
- `0`: absence of heart disease.

The project compares five classification algorithms using multiple evaluation metrics. Using more than accuracy helps assess model performance when evaluating clinical outcomes where false-positive and false-negative predictions carry distinct medical implications.

## b. Dataset Description

**Dataset:** Heart Disease  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/45/heart+disease

The Heart Disease dataset contains clinical patient attributes and diagnostic indicators. The target column is `target`, which indicates whether a patient has cardiovascular disease. The dataset contains 1,025 records and 13 predictive input features (such as age, sex, chest pain type, resting blood pressure, cholesterol, and maximum heart rate), in addition to the target column.

Preprocessing handles categorical variable encoding and numerical scaling required by distance-based and linear models. The same fitted preprocessing steps are applied consistently during both training and evaluation.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/yogalakshmidt-dev/Heart-Disease-Classification

The repository includes the application, training code, dependency list, test CSV (`test_data.csv`), saved trained models, preprocessing artifacts, and this README. Confirm that all required files and model artifacts are committed and visible in the repository before submission.

**Streamlit Application:** https://heart-disease-classification-05743.streamlit.app/

## d. Models Used

The five classification models evaluated in this project are:

1. **Logistic Regression** — a probabilistic linear classification model.
2. **Decision Tree** — a rule-based model that captures nonlinear threshold relationships.
3. **K-Nearest Neighbors (KNN)** — predicts a class based on nearby training examples and is sensitive to feature scaling.
4. **Naive Bayes** — a probabilistic classifier based on Bayes' theorem and conditional-independence assumptions.
5. **Support Vector Machine (SVM)** — learns an optimal decision boundary between classes using kernel methods.

### Model Comparison Table

The following values are taken from the `metrics_summary.json` file generated during evaluation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7951 | 0.8440 | 0.6471 | 0.5789 | 0.6111 | 0.4740 |
| Decision Tree | 0.7366 | 0.7546 | 0.5294 | 0.4737 | 0.5000 | 0.3229 |
| KNN | 0.8000 | 0.7887 | 0.6667 | 0.5614 | 0.6095 | 0.4796 |
| Naive Bayes | 0.8098 | 0.8372 | 0.6552 | 0.6667 | 0.6609 | 0.5287 |
| Support Vector Machine | 0.8098 | 0.8222 | 0.6957 | 0.5614 | 0.6214 | 0.5013 |

### Evaluation Metrics

- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** area under the ROC curve; measures discrimination between positive and negative classes across thresholds.
- **Precision:** proportion of predicted positive cases that are truly positive.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient; summarizes agreement between predicted and actual classes using all four confusion-matrix outcomes. It ranges from -1 to +1.

### Results and Best-Model Observation

From the supplied comparison results:

- **Highest Accuracy:** Naive Bayes / Support Vector Machine (0.8098)
- **Highest AUC:** Logistic Regression (0.8440)
- **Highest Precision:** Support Vector Machine (0.6957)
- **Highest Recall:** Naive Bayes (0.6667)
- **Highest F1:** Naive Bayes (0.6609)
- **Highest MCC:** Naive Bayes (0.5287)

A model should not be selected using accuracy alone. In medical diagnostic tasks, **Recall** and **MCC** are critical because failing to identify a patient with heart disease (false negative) carries severe consequences. Based on the evaluation pipeline, **Naive Bayes** is the top-performing model overall, achieving the highest Recall (0.6667), F1-score (0.6609), and MCC (0.5287).

To run the application locally, install the dependencies from `requirements.txt` and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Links

- **GitHub Repository:** https://github.com/yogalakshmidt-dev/Heart-Disease-Classification
- **Streamlit Application:** https://heart-disease-classification-05743.streamlit.app/
- **Dataset:** https://archive.ics.uci.edu/dataset/45/heart+disease

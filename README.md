# BITS Pilani - M.Tech (AIML) Machine Learning Assignment 2
**Work Integrated Learning Programmes Division**  
**Course**: Machine Learning  
**Assignment**: Assignment - 2 (Marks: 15)  
**Submission Deadline**: 08-Nov-2026  

---

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification models on a high-dimensional clinical dataset to predict the presence of cardiovascular disease. Heart disease remains the leading cause of mortality worldwide, and timely, accurate algorithmic screening can significantly assist clinical decision-making. 

In this work:
1. Five distinct machine learning classification models are implemented, trained, and evaluated:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbor (KNN) Classifier
   - Naive Bayes Classifier (Gaussian)
   - Support Vector Machine (SVM)
2. All 5 models are evaluated across 6 standardized performance metrics without retraining on the test split:
   - Accuracy
   - AUC Score (Area Under the ROC Curve)
   - Precision
   - Recall (Sensitivity)
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
3. An interactive Streamlit web application is developed and deployed to enable interactive model selection, test data upload, confusion matrix visualization, and real-time comparative metric inspection.

---

## b. Dataset Description [1 Mark]
The dataset selected for this study is the **Heart Disease Classification Dataset**, sourced from the **UCI Machine Learning Repository** and **Kaggle**.

### Dataset Specifications:
- **Total Instance Size**: 1,025 instances (Meets minimum requirement of ≥ 500 instances)
- **Feature Size**: 13 predictive clinical features + 1 target variable (Meets minimum requirement of ≥ 12 features)
- **Target Variable**: `target` (Binary Classification: `0` = Absence of heart disease, `1` = Presence of heart disease)
- **Train-Test Split**: Stratified 80% Training (820 instances) and 20% Testing (205 instances, saved as `test_data.csv`).

### Feature Dictionary:
| # | Feature Name | Data Type | Value Range / Categories | Description |
|---|---|---|---|---|
| 1 | `age` | Numeric | 29 – 77 years | Age of the patient in years |
| 2 | `sex` | Categorical | 0 = Female, 1 = Male | Biological sex of the patient |
| 3 | `cp` | Categorical | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic | Chest pain type experienced |
| 4 | `trestbps` | Numeric | 94 – 200 mm Hg | Resting blood pressure on admission to the hospital |
| 5 | `chol` | Numeric | 126 – 564 mg/dl | Serum cholesterol measurement |
| 6 | `fbs` | Categorical | 0 = False (≤ 120 mg/dl), 1 = True (> 120 mg/dl) | Fasting blood sugar level |
| 7 | `restecg` | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy | Resting electrocardiographic results |
| 8 | `thalach` | Numeric | 71 – 202 bpm | Maximum heart rate achieved during exercise |
| 9 | `exang` | Categorical | 0 = No, 1 = Yes | Exercise-induced angina |
| 10 | `oldpeak` | Numeric | 0.0 – 6.2 | ST depression induced by exercise relative to rest |
| 11 | `slope` | Categorical | 0: Upsloping, 1: Flat, 2: Downsloping | The slope of the peak exercise ST segment |
| 12 | `ca` | Discrete | 0 – 3 | Number of major vessels colored by fluoroscopy |
| 13 | `thal` | Categorical | 1: Normal, 2: Fixed defect, 3: Reversible defect | Thallium stress test result |
| 14 | `target` | Categorical | 0 = No Disease, 1 = Heart Disease | Ground truth diagnosis target column |

---

## c. GitHub Repository Link [1 Mark]
- **GitHub Repository**: `https://github.com/yogalakshmidt/bits-aiml-ml-assignment-2` *(Replace with your GitHub repository URL)*
- **Live Streamlit Community Cloud URL**: `https://heart-disease-classifier-bits.streamlit.app` *(Replace with your deployed Streamlit Cloud URL)*

### Repository Structure:
```text
project-folder/
│-- app.py                     # Interactive Streamlit Web Application
│-- train_models.py            # Model training & artifact generation pipeline
│-- requirements.txt           # Python dependency specifications
│-- README.md                  # Comprehensive documentation & submission report
│-- test_data.csv              # Test dataset (205 samples with features + target)
│-- model/                     # Pre-trained saved models & preprocessing artifacts
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- svm.joblib
    │-- scaler.joblib
    │-- feature_names.joblib
    │-- metrics_summary.json
    │-- confusion_matrices.json
    │-- classification_reports.json
    │-- dataset_metadata.json
```

---

## d. Models Used & Comparison Table [5 Marks]
All 5 classification models were trained on the training partition (820 instances) with standard scaling applied to distance/linear sensitive algorithms. Evaluation was performed on the independent held-out `test_data.csv` (205 instances).

### Mandatory Evaluation Metrics Comparison Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.7951 | 0.8440 | 0.6471 | 0.5789 | 0.6111 | 0.4740 |
| **Decision Tree** | 0.7366 | 0.7546 | 0.5294 | 0.4737 | 0.5000 | 0.3229 |
| **KNN** | 0.8000 | 0.7887 | 0.6667 | 0.5614 | 0.6095 | 0.4796 |
| **Naive Bayes** | 0.8098 | 0.8372 | 0.6552 | 0.6667 | 0.6609 | 0.5287 |
| **Support Vector Machine** | 0.8098 | 0.8222 | 0.6957 | 0.5614 | 0.6214 | 0.5013 |

---

## e. Observations on Model Performance [3 Marks]

| ML Model Name | Observation about model performance |
|:---|:---|
| **Logistic Regression** | Achieved the **highest AUC score (0.8440)** among all five classifiers, demonstrating exceptional overall discrimination across varying decision thresholds. With an accuracy of 79.51% and an F1 score of 0.6111, it proves to be a well-calibrated, highly reliable baseline with strong clinical interpretability. |
| **Decision Tree** | Yielded an accuracy of 73.66% and an MCC of 0.3229. While decision trees offer maximum human interpretability through hierarchical if-else decision rules, individual unpruned/shallow trees are sensitive to local variations and lack the smooth boundary regularization of linear and kernel methods. |
| **KNN** | Reached 80.00% accuracy and an MCC of 0.4796 utilizing distance-weighted nearest neighbors ($k=7$) on standard-scaled features. It handles local non-linear patient sub-clusters effectively, though inference requires retaining training instances. |
| **Naive Bayes** | Tied for the **highest Accuracy (80.98%)** and secured the **highest Recall (0.6667)**, **highest F1-score (0.6609)**, and **highest MCC (0.5287)**. Despite the theoretical assumption of conditional feature independence, Gaussian Naive Bayes excels on this biometric feature set, effectively capturing the marginal likelihoods of disease indicators. |
| **Support Vector Machine** | Tied for the **highest Accuracy (80.98%)** and delivered the **highest Precision (0.6957)** with an AUC of 0.8222. The non-linear Radial Basis Function (RBF) kernel mapped the 13 clinical variables into a higher-dimensional space where separation of complex risk factors was maximized. |

### Overall, Winner for your dataset?
**Overall Winner: Naive Bayes (GaussianNB)** *(with Support Vector Machine as close runner-up)*

#### Justification:
In clinical healthcare applications such as heart disease diagnosis, **Recall (Sensitivity)** is the single most vital metric because the clinical penalty of a **False Negative** (failing to identify an individual with cardiovascular disease) is significantly worse than a False Positive. 
- **Naive Bayes achieved the highest Recall (0.6667)**, successfully identifying two-thirds of positive disease cases in the test set.
- It also achieved the **highest Matthews Correlation Coefficient (MCC = 0.5287)**, which is the most robust single measure of binary classification quality on imbalanced/stratified datasets.
- Additionally, it tied for **highest overall Accuracy (80.98%)** and achieved the **highest F1 Score (0.6609)**.
- For scenarios where false positives must be minimized (e.g. expensive invasive follow-ups), **Support Vector Machine** is the preferred model, achieving the highest Precision (0.6957).

---

## Step 6: How to Deploy on Streamlit Community Cloud (Step-by-Step)
1. Push this project to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Complete ML Assignment 2: Heart Disease Classification"
   git remote add origin https://github.com/<your-username>/bits-aiml-ml-assignment-2.git
   git push -u origin main
   ```
2. Navigate to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Sign in using your GitHub account credentials.
4. Click **New App**.
5. Select your repository: `<your-username>/bits-aiml-ml-assignment-2`.
6. Select Branch: `main`.
7. Select Main file path: `app.py`.
8. Click **Deploy!**

---

## Verification & Execution on BITS Virtual Lab [1 Mark]
Per assignment instructions:
1. Open BITS Virtual Lab environment.
2. Clone repository or upload the folder.
3. Run training script:
   ```bash
   python3 train_models.py
   ```
4. Launch Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Take a screenshot showing the terminal execution and Streamlit browser window to include in your final single-PDF submission.

# 💰 Loan Eligibility Prediction

This project aims to predict whether a loan application will be approved based on various applicant features like income, education, credit history, and more. The goal is to assist financial institutions in automating and improving the decision-making process for loan approvals.

---

## 📂 Dataset Overview

- **Source**: Loan Prediction dataset (commonly found on platforms like Analytics Vidhya/Kaggle)
- **Total Records**: 614
- **Target Variable**: `Loan_Status` (Y = Approved, N = Not Approved)

### 🔢 Columns

| Column Name         | Description |
|---------------------|-------------|
| `Loan_ID`           | Unique identifier for each loan application |
| `Gender`            | Applicant's gender (Male/Female) |
| `Married`           | Marital status (Yes/No) |
| `Dependents`        | Number of dependents |
| `Education`         | Education level (Graduate/Not Graduate) |
| `Self_Employed`     | Employment status (Yes/No) |
| `ApplicantIncome`   | Monthly income of the applicant |
| `CoapplicantIncome` | Monthly income of co-applicant |
| `LoanAmount`        | Loan amount in thousands |
| `Loan_Amount_Term`  | Term of the loan in months |
| `Credit_History`    | Credit history (1.0 = good, 0.0 = bad) |
| `Property_Area`     | Urban, Semiurban, or Rural |
| `Loan_Status`       | (Target) Loan approved (Y) or not (N) |

---

## 🔍 Exploratory Data Analysis (EDA)

- Examined distribution of income, loan amount, loan term, and credit history
- Visualized approval rates across gender, education, self-employment, and property area
- Checked and imputed missing values:
  - Used mode/median imputation depending on the column type
  - Detected and handled outliers (e.g., income)

---

## 🛠️ Preprocessing

- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Imputed missing values
- Scaled numerical features when necessary
- Converted `Loan_Status` to binary (Y = 1, N = 0)
- Created a new `Total_Income` feature to combine applicant and coapplicant income

---

## 🤖 Modeling

- Split data into training and testing sets
- Models used:
  - Logistic Regression
  - Random Forest (tuned via GridSearchCV)
- Predictions made on test set
- Evaluated using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - ROC Curve and AUC Score

---

## 📊 Evaluation

| Metric       | Value           |
|--------------|-----------------|
| Accuracy     | 81.3%           |
| Precision    | 0.82            |
| Recall       | 0.85            |
| F1-score     | 0.83            |
| AUC Score    | 0.88            |

> 📌 These results may vary based on preprocessing steps and final model selection.

---

## 📈 ROC Curve

- ROC curve plotted using `roc_curve`
- AUC calculated with `roc_auc_score`
- Probabilities obtained using `.predict_proba()` on the test set

---

## 🧰 Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## ✅ Conclusion

This project delivers an end-to-end pipeline for predicting loan eligibility using machine learning. It demonstrates feature engineering, handling of missing data, model training, hyperparameter tuning, and evaluation through common metrics.

---

## 🚀 Future Enhancements

- Use ensemble methods like XGBoost or CatBoost
- Address class imbalance with SMOTE
- Deploy as a web app with Streamlit or Flask
- Implement explainability (SHAP or LIME)


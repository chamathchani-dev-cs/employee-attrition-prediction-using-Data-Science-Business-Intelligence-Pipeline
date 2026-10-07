# Employee Attrition Prediction

CSC 630 – Data Science, Business Intelligence and Big Data Analytics (MSc in Computer Science), final assignment.

Predicts employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset (1,470 records, 35 attributes).

## Contents
- `CSC630_Employee_Attrition_Prediction.ipynb` – full analysis: preprocessing, exploration, and feature engineering, model comparison and class-imbalance optimisation (Logistic Regression, KNN, Decision Tree, Random Forest)
- `presentation/` – assignment presentation slides

## Key Results
- Dataset: IBM HR Analytics, 1,470 employees, ~16% attrition (imbalanced)
- Models compared: Logistic Regression, Decision Tree, Random Forest, KNN
- Initial accuracy was misleading: Random Forest and KNN scored ~84% but caught only 13-15% of leavers
- After class-weighted training, the final Logistic Regression model achieved 65.9% recall, 36.9% precision and 0.811 AUC
- Key attrition drivers: overtime (30.5% vs 10.4%), frequent business travel, job role (Sales Representatives ~40%), lower income, single marital status

## How to run
1. Download the dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) from Kaggle and place it next to the notebook.
2. `pip install -r requirements.txt`
3. `jupyter notebook` and open the `.ipynb` file, then Run All.

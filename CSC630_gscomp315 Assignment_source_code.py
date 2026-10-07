import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')  # adjust filename if different
df.shape

# ---- cell ----

df.head()

# ---- cell ----

df.info()

# ---- cell ----

df.isnull().sum()

# ---- cell ----

plt.figure(figsize=(14,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Value Heatmap (Yellow = Missing)')
plt.show()

# ---- cell ----

df.duplicated().sum()

# ---- cell ----

duplicate_count = df.duplicated().sum()
unique_count = len(df) - duplicate_count

plt.figure(figsize=(5,4))
plt.bar(['Unique Rows', 'Duplicate Rows'], [unique_count, duplicate_count], color=['steelblue', 'crimson'])
plt.title('Unique vs Duplicate Rows')
plt.ylabel('Count')
for i, v in enumerate([unique_count, duplicate_count]):
    plt.text(i, v + 10, str(v), ha='center')
plt.show()

# ---- cell ----

for col in ['EmployeeCount', 'Over18', 'StandardHours']:
    print(col, ':', df[col].unique())

# ---- cell ----

cols_to_check = ['EmployeeCount', 'Over18', 'StandardHours']
unique_counts = [df[col].nunique() for col in cols_to_check]

plt.figure(figsize=(6,4))
plt.bar(cols_to_check, unique_counts, color='orange')
plt.title('Number of Unique Values per Column')
plt.ylabel('Unique Value Count')
for i, v in enumerate(unique_counts):
    plt.text(i, v + 0.05, str(v), ha='center')
plt.ylim(0, 2)
plt.show()

# ---- cell ----

constant_check = pd.DataFrame({
    'Column': cols_to_check,
    'Unique Values': [df[col].unique() for col in cols_to_check],
    'Number of Unique Values': [df[col].nunique() for col in cols_to_check]
})
constant_check

# ---- cell ----

df['Attrition'].value_counts()
df['Attrition'].value_counts(normalize=True) * 100

# ---- cell ----

plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Attrition (Target Variable) Distribution')
plt.xlabel('Attrition')
plt.ylabel('Number of Employees')

# add count labels on top of bars
for i, count in enumerate(df['Attrition'].value_counts().reindex(['No','Yes'])):
    plt.text(i, count + 10, str(count), ha='center')

plt.show()

# ---- cell ----

attrition_pct = df['Attrition'].value_counts(normalize=True) * 100

plt.figure(figsize=(6,4))
attrition_pct.plot(kind='bar', color=['steelblue','crimson'])
plt.title('Attrition Distribution (%)')
plt.xlabel('Attrition')
plt.ylabel('Percentage')
plt.xticks(rotation=0)

for i, v in enumerate(attrition_pct):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.show()

# ---- cell ----

attrition_pct = df['Attrition'].value_counts(normalize=True) * 100

plt.figure(figsize=(5,5))
plt.pie(attrition_pct, labels=attrition_pct.index, autopct='%1.1f%%', 
        colors=['steelblue','crimson'], startangle=90)
plt.title('Attrition Proportion (%)')
plt.show()

# ---- cell ----

continuous_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                    'PercentSalaryHike', 'TotalWorkingYears', 
                    'TrainingTimesLastYear', 'YearsAtCompany', 
                    'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                    'YearsWithCurrManager']

ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

categorical_cols = ['Attrition', 'BusinessTravel', 'Department', 
                    'EducationField', 'Gender', 'JobRole', 
                    'MaritalStatus', 'OverTime']

print("Continuous:", len(continuous_cols))
print("Ordinal:", len(ordinal_cols))
print("Categorical:", len(categorical_cols))

# ---- cell ----

group_counts = {
    'Continuous': len(continuous_cols),
    'Ordinal': len(ordinal_cols),
    'Categorical': len(categorical_cols)
}

plt.figure(figsize=(6,4))
plt.bar(group_counts.keys(), group_counts.values(), color=['steelblue','orange','mediumpurple'])
plt.title('Column Type Breakdown')
plt.ylabel('Number of Columns')

for i, (k, v) in enumerate(group_counts.items()):
    plt.text(i, v + 0.3, str(v), ha='center')

plt.show()

# ---- cell ----

plt.figure(figsize=(5,5))
plt.pie(group_counts.values(), labels=group_counts.keys(), autopct='%1.1f%%',
        colors=['steelblue','orange','mediumpurple'])
plt.title('Proportion of Column Types')
plt.show()

# ---- cell ----

fig, axes = plt.subplots(4, 4, figsize=(18,16))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='steelblue')
    axes[i].set_title(col)

# hide unused subplots
for j in range(len(continuous_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ---- cell ----

df[continuous_cols].hist(figsize=(18,14), bins=20, color='teal')
plt.tight_layout()
plt.show()

# ---- cell ----

fig, axes = plt.subplots(3, 3, figsize=(18,14))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i], color='mediumpurple')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=30)

for j in range(len(categorical_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ---- cell ----

plt.figure(figsize=(7,5))
sns.scatterplot(x='TotalWorkingYears', y='MonthlyIncome', data=df, alpha=0.5, color='steelblue')
plt.title('Numerical-Numerical: Total Working Years vs Monthly Income')
plt.show()

print(df[['TotalWorkingYears','MonthlyIncome']].corr())

# ---- cell ----

pd.crosstab(df['OverTime'], df['Attrition'])

# ---- cell ----

plt.figure(figsize=(6,4))
sns.countplot(x='OverTime', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Categorical-Categorical: OverTime vs Attrition')
plt.show()

# ---- cell ----

plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette=['steelblue','crimson'])
plt.title('Numerical-Categorical: Monthly Income by Attrition')
plt.show()

# ---- cell ----

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, hue='Attrition', palette=['steelblue','crimson'], legend=False)

# ---- cell ----

avg_years = df.groupby('JobRole')['YearsAtCompany'].mean().sort_values()

plt.figure(figsize=(8,5))
avg_years.plot(kind='barh', color='teal')
plt.title('Categorical-Numerical: Average Years at Company by Job Role')
plt.xlabel('Average Years at Company')
plt.show()


# ---- cell ----

attrition_by_role = df.groupby('JobRole')['Attrition'].apply(lambda x: (x=='Yes').mean() * 100).sort_values(ascending=False)

plt.figure(figsize=(8,5))
attrition_by_role.plot(kind='barh', color='crimson')
plt.title('Attrition Rate (%) by Job Role')
plt.xlabel('Attrition Rate (%)')
plt.show()

# ---- cell ----

plt.figure(figsize=(7,4))
sns.countplot(x='BusinessTravel', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Business Travel Frequency vs Attrition')
plt.show()

# ---- cell ----

df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])

# ---- cell ----

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                     'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())

# ---- cell ----

fig, axes = plt.subplots(4, 2, figsize=(14,16))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color='mediumpurple')
    axes[i].set_title(f'Frequency: {col}')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=30)

# hide unused subplot (since 7 columns won't fill an 8-slot grid)
for j in range(len(categorical_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ---- cell ----

df_onehot = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 
                                          'EducationField', 'Gender', 
                                          'JobRole', 'MaritalStatus', 
                                          'OverTime'], drop_first=True)
df_onehot.head()

# ---- cell ----

df_freq = df.copy()

for col in categorical_cols:
    freq_map = df_freq[col].value_counts().to_dict()
    df_freq[col + '_Freq'] = df_freq[col].map(freq_map)

df_freq[['JobRole', 'JobRole_Freq', 'Department', 'Department_Freq']].head()

# ---- cell ----

# Version A: One-Hot Encoding (all categorical columns)
df_onehot = df.copy()
df_onehot['Attrition'] = df_onehot['Attrition'].map({'Yes': 1, 'No': 0})
df_onehot = pd.get_dummies(df_onehot, columns=categorical_cols, drop_first=True)

print("One-Hot Encoded shape:", df_onehot.shape)

# ---- cell ----

# Version B: Frequency Encoding (all categorical columns)
df_freq = df.copy()
df_freq['Attrition'] = df_freq['Attrition'].map({'Yes': 1, 'No': 0})

for col in categorical_cols:
    freq_map = df_freq[col].value_counts().to_dict()
    df_freq[col] = df_freq[col].map(freq_map)

print("Frequency Encoded shape:", df_freq.shape)

# ---- cell ----

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_encoding(dataframe, name):
    X = dataframe.drop('Attrition', axis=1)
    y = dataframe['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Number of features: {X.shape[1]}\n")
    
    return acc, auc

acc_onehot, auc_onehot = evaluate_encoding(df_onehot, "One-Hot Encoding")
acc_freq, auc_freq = evaluate_encoding(df_freq, "Frequency Encoding")

# ---- cell ----

comparison = pd.DataFrame({
    'Encoding Method': ['One-Hot', 'Frequency'],
    'Accuracy': [acc_onehot, acc_freq],
    'AUC': [auc_onehot, auc_freq]
})

comparison.set_index('Encoding Method').plot(kind='bar', figsize=(7,5), color=['steelblue','crimson'])
plt.title('Encoding Method Comparison: Accuracy vs AUC')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.show()

comparison

# ---- cell ----

def attrition_rate(condition):
    subset = df[condition]
    return round((subset['Attrition'] == 'Yes').mean() * 100, 1)

diagnostic_summary = pd.DataFrame({
    'Risk Factor': [
        'OverTime = Yes', 'OverTime = No',
        'BusinessTravel = Frequent', 'BusinessTravel = Rare/None',
        'JobRole = Sales Representative', 'JobRole = Research Director',
        'MaritalStatus = Single', 'MaritalStatus = Married/Divorced'
    ],
    'Attrition Rate (%)': [
        attrition_rate(df['OverTime'] == 'Yes'),
        attrition_rate(df['OverTime'] == 'No'),
        attrition_rate(df['BusinessTravel'] == 'Travel_Frequently'),
        attrition_rate(df['BusinessTravel'] != 'Travel_Frequently'),
        attrition_rate(df['JobRole'] == 'Sales Representative'),
        attrition_rate(df['JobRole'] == 'Research Director'),
        attrition_rate(df['MaritalStatus'] == 'Single'),
        attrition_rate(df['MaritalStatus'] != 'Single')
    ]
})

diagnostic_summary

# ---- cell ----

plt.figure(figsize=(10,6))
sns.barplot(x='Attrition Rate (%)', y='Risk Factor', data=diagnostic_summary, palette='Reds_r')
plt.title('Diagnostic Summary: Attrition Rate by Key Risk Factor')
plt.xlabel('Attrition Rate (%)')
plt.show()

# ---- cell ----

df['AgeGroup'] = pd.cut(df['Age'], 
                          bins=[18, 25, 35, 45, 55, 65], 
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65'])

df[['Age', 'AgeGroup']].head(10)

# ---- cell ----

sns.countplot(x='AgeGroup', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Edited Feature: Attrition by Age Group')
plt.show()

# ---- cell ----

df['HighStrain'] = ((df['OverTime'] == 'Yes') | (df['BusinessTravel'] == 'Travel_Frequently')).astype(int)

df[['OverTime', 'BusinessTravel', 'HighStrain']].head(10)

# ---- cell ----

sns.countplot(x='HighStrain', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('New Feature: Attrition by High Strain Indicator')
plt.xticks([0,1], ['Low Strain', 'High Strain'])
plt.show()

# ---- cell ----

df['IncomePerYearWorked'] = df['MonthlyIncome'] / (df['TotalWorkingYears'].replace(0, 1))

df[['MonthlyIncome', 'TotalWorkingYears', 'IncomePerYearWorked']].head(10)

# ---- cell ----

sns.boxplot(x='Attrition', y='IncomePerYearWorked', data=df, palette=['steelblue','crimson'])
plt.title('Multi-Layer Feature: Income per Year Worked by Attrition')
plt.show()

# ---- cell ----

# Correlation between the multi-layer feature and its components
correlation_check = df[['IncomePerYearWorked', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']].corr()
correlation_check

# ---- cell ----

plt.figure(figsize=(6,4))
sns.heatmap(correlation_check, annot=True, cmap='coolwarm')
plt.title('Correlation: Multi-Layer Feature vs Source Features')
plt.show()

# ---- cell ----

# Simulate: same MonthlyIncome, different TotalWorkingYears
example_income = 5000
example_years = [1, 5, 10, 20, 30]

simulation = pd.DataFrame({
    'TotalWorkingYears': example_years,
    'MonthlyIncome': [example_income]*len(example_years),
    'IncomePerYearWorked': [example_income/y for y in example_years]
})
simulation

# ---- cell ----

plt.plot(simulation['TotalWorkingYears'], simulation['IncomePerYearWorked'], marker='o', color='purple')
plt.title('Effect of TotalWorkingYears on IncomePerYearWorked (Fixed Income = $5000)')
plt.xlabel('TotalWorkingYears')
plt.ylabel('IncomePerYearWorked')
plt.show()

# ---- cell ----

df['DistanceCategory'] = pd.cut(df['DistanceFromHome'],
                                   bins=[0, 5, 10, 20, 30],
                                   labels=['Very Close', 'Close', 'Moderate', 'Far'])

df[['DistanceFromHome', 'DistanceCategory']].head(10)

# ---- cell ----

sns.countplot(x='DistanceCategory', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Edited Feature: Attrition by Distance Category')
plt.show()

# ---- cell ----

df['IsSingle'] = (df['MaritalStatus'] == 'Single').astype(int)
df[['MaritalStatus', 'IsSingle']].head(10)

# ---- cell ----

plt.figure(figsize=(6,4))
sns.countplot(x='IsSingle', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('New Feature: Attrition by Marital Status (Single vs Not Single)')
plt.xticks([0,1], ['Not Single', 'Single'])
plt.show()

# ---- cell ----

pd.crosstab(df['IsSingle'], df['Attrition'])

# ---- cell ----

df['OvertimeLowSatisfaction'] = ((df['OverTime'] == 'Yes') & (df['JobSatisfaction'] <= 2)).astype(int)

pd.crosstab(df['OvertimeLowSatisfaction'], df['Attrition'])

# ---- cell ----

sns.countplot(x='OvertimeLowSatisfaction', hue='Attrition', data=df, palette=['steelblue','crimson'])
plt.title('Multi-Layer Feature: Overtime + Low Satisfaction vs Attrition')
plt.xticks([0,1], ['Not Both', 'Overtime + Low Satisfaction'])
plt.show()

# ---- cell ----

# Fix other satisfaction scores, vary WorkLifeBalance
fixed_env, fixed_job, fixed_rel = 3, 3, 3
worklife_values = [1, 2, 3, 4]

impact_sim = pd.DataFrame({
    'WorkLifeBalance': worklife_values,
    'TotalSatisfaction': [fixed_env + fixed_job + fixed_rel + w for w in worklife_values]
})
impact_sim

# ---- cell ----

plt.plot(impact_sim['WorkLifeBalance'], impact_sim['TotalSatisfaction'], marker='o', color='darkorange')
plt.title('Effect of WorkLifeBalance on TotalSatisfaction (Other Scores Fixed)')
plt.xlabel('WorkLifeBalance Rating')
plt.ylabel('TotalSatisfaction Score')
plt.show()

# ---- cell ----

# Final encoding on the enriched dataframe (including all engineered features)
df_final = df.copy()
df_final['Attrition'] = df_final['Attrition'].map({'Yes': 1, 'No': 0})

df_final_encoded = pd.get_dummies(df_final, drop_first=True)
df_final_encoded.shape

# ---- cell ----

plt.figure(figsize=(9,7))
importances.head(20).plot(kind='barh', color='mediumseagreen')
plt.title('Top 20 Feature Importances (Including Engineered Features)')
plt.gca().invert_yaxis()
plt.show()

# ---- cell ----

X = df_final_encoded.drop('Attrition', axis=1)
y = df_final_encoded['Attrition']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ---- cell ----

shape_summary = pd.DataFrame({
    'Component': ['Total Records', 'Feature Columns', 'Target Variable'],
    'Count': [X.shape[0], X.shape[1], 1]
})
shape_summary

# ---- cell ----

plt.figure(figsize=(6,4))
plt.bar(['Records', 'Features'], [X.shape[0], X.shape[1]], color=['steelblue','orange'])
plt.title('Dataset Dimensions for Model Training')
plt.ylabel('Count')

for i, v in enumerate([X.shape[0], X.shape[1]]):
    plt.text(i, v + 10, str(v), ha='center')

plt.show()

# ---- cell ----

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
print("\nTraining target distribution:\n", y_train.value_counts(normalize=True))
print("\nTest target distribution:\n", y_test.value_counts(normalize=True))

# ---- cell ----

import numpy as np

split_data = pd.DataFrame({
    'No (0)': [0.838435, 0.840136],
    'Yes (1)': [0.161565, 0.159864]
}, index=['Training Set', 'Test Set'])

split_data.plot(kind='bar', stacked=True, figsize=(7,5), color=['steelblue','crimson'])
plt.title('Class Distribution: Training vs Test Set')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.show()

split_data

# ---- cell ----

split_data.plot(kind='bar', figsize=(7,5), color=['steelblue','crimson'])
plt.title('Class Distribution: Training vs Test Set')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.show()

# ---- cell ----

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")

# ---- cell ----

from sklearn.metrics import accuracy_score

train_accuracies = {}

for name, model in trained_models.items():
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    train_accuracies[name] = train_acc

plt.figure(figsize=(8,5))
bars = plt.bar(train_accuracies.keys(), train_accuracies.values(), 
                color=['steelblue','orange','seagreen','crimson'])
plt.title('Training Set Accuracy by Model (Preview Only)')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
plt.xticks(rotation=15)

for bar, acc in zip(bars, train_accuracies.values()):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.3f}', ha='center')

plt.show()

# ---- cell ----

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- cell ----

# Pick a feature with a wide range for a clear visual difference
sample_feature = 'MonthlyIncome'
feature_index = list(X_train.columns).index(sample_feature)

fig, axes = plt.subplots(1, 2, figsize=(12,4))

# Before scaling
axes[0].hist(X_train[sample_feature], bins=30, color='steelblue')
axes[0].set_title(f'Before Scaling: {sample_feature}')
axes[0].set_xlabel('Original Value')

# After scaling
axes[1].hist(X_train_scaled[:, feature_index], bins=30, color='crimson')
axes[1].set_title(f'After Scaling: {sample_feature}')
axes[1].set_xlabel('Scaled Value (Standardized)')

plt.tight_layout()
plt.show()

# ---- cell ----

compare_features = ['MonthlyIncome', 'Age', 'DistanceFromHome']
compare_indices = [list(X_train.columns).index(f) for f in compare_features]

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Before scaling - boxplot shows very different ranges
X_train[compare_features].boxplot(ax=axes[0])
axes[0].set_title('Before Scaling (Different Ranges)')

# After scaling - boxplot shows comparable ranges
pd.DataFrame(X_train_scaled[:, compare_indices], columns=compare_features).boxplot(ax=axes[1])
axes[1].set_title('After Scaling (Standardized Ranges)')

plt.tight_layout()
plt.show()

# ---- cell ----

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")

# ---- cell ----

from sklearn.metrics import accuracy_score

train_accuracies = {}

for name, model in trained_models.items():
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    train_accuracies[name] = train_acc

plt.figure(figsize=(6,4))
plt.bar(train_accuracies.keys(), train_accuracies.values(), color=['steelblue','seagreen'])
plt.title('Training Set Accuracy by Model (Preview)')
plt.ylabel('Accuracy')
plt.ylim(0,1)

for i, (name, acc) in enumerate(train_accuracies.items()):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')

plt.show()

# ---- cell ----

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, roc_auc_score, confusion_matrix, classification_report)

results = []

for name, model in trained_models.items():
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:,1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, proba)
    })

results_df = pd.DataFrame(results).set_index('Model')
results_df

# ---- cell ----

results_df.plot(kind='bar', figsize=(12,6), colormap='viridis')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0,1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.show()

# ---- cell ----

fig, axes = plt.subplots(2, 2, figsize=(12,10))
axes = axes.flatten()

for i, (name, model) in enumerate(trained_models.items()):
    preds = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    axes[i].set_title(f'{name} - Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ---- cell ----

overfit_check = []

for name, model in trained_models.items():
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    overfit_check.append({'Model': name, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc, 'Gap': train_acc - test_acc})

overfit_df = pd.DataFrame(overfit_check)
overfit_df

# ---- cell ----

overfit_df.set_index('Model')[['Train Accuracy','Test Accuracy']].plot(kind='bar', figsize=(9,5), color=['steelblue','crimson'])
plt.title('Training vs Test Accuracy (Overfitting Check)')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
plt.xticks(rotation=15)
plt.show()

# ---- cell ----

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

optimized_models = {
    'Logistic Regression (Balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(class_weight='balanced', random_state=42, max_depth=8)
}

optimized_results = []

for name, model in optimized_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:,1]
    
    optimized_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, proba)
    })

optimized_df = pd.DataFrame(optimized_results).set_index('Model')
optimized_df

# ---- cell ----

comparison_optimization = pd.concat([
    results_df.loc[['Logistic Regression','Random Forest']],
    optimized_df
])

comparison_optimization[['Recall','Precision','F1-Score']].plot(kind='bar', figsize=(10,6))
plt.title('Before vs After Optimization: Recall Improvement')
plt.ylabel('Score')
plt.xticks(rotation=30, ha='right')
plt.show()

# ---- cell ----

final_model = trained_models_balanced['Logistic Regression (Balanced)'] if 'trained_models_balanced' in dir() else optimized_models['Logistic Regression (Balanced)']

coefficients = pd.Series(final_model.coef_[0], index=X.columns).sort_values()

plt.figure(figsize=(9,10))
coefficients.tail(15).plot(kind='barh', color='crimson')
plt.title('Top 15 Features Increasing Attrition Risk (Logistic Regression Coefficients)')
plt.xlabel('Coefficient (Higher = Increases Attrition Likelihood)')
plt.show()

# ---- cell ----

plt.figure(figsize=(9,8))
coefficients.head(15).plot(kind='barh', color='seagreen')
plt.title('Top 15 Features Decreasing Attrition Risk (Logistic Regression Coefficients)')
plt.xlabel('Coefficient (More Negative = Reduces Attrition Likelihood)')
plt.show()

# ---- cell ----


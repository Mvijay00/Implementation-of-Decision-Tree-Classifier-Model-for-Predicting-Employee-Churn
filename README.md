# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess data: Import the dataset, drop unnecessary columns (like ID), create the binary target variable "Attrition" based on tenure and salary conditions, and encode categorical text columns into numeric values.

2.Split data: Separate features and target variable, then split the dataset into training and testing subsets to prepare for model training and evaluation.

3.Train the classifier: Initialize a Decision Tree Classifier with a maximum depth constraint (to prevent overfitting) and train it on the training data.

4.Evaluate the model: Make predictions on the test dataset, compute accuracy, and display the most important features influencing the classification.

5.Visualize the decision tree: Plot the trained decision tree structure to show the decision rules and feature splits that lead to attrition predictions.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIJAYARAGHAVAN M
RegisterNumber: 25017872 
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
data = pd.read_csv("employee_churn_dataset.csv")
print(data.head())

# Drop ID column
if 'ID' in data.columns:
    data = data.drop(['ID'], axis=1)

# Create target variable
data['Attrition'] = ((data['Tenure'] < 2) & (data['Salary'] < data['Salary'].median())).astype(int)
print(f"Created Attrition: {data['Attrition'].value_counts().to_dict()}")

# Encode text columns
le = LabelEncoder()
for col in ['Gender', 'Education Level', 'Marital Status', 'Job Role', 'Department', 'Work Location']:
    data[col] = le.fit_transform(data[col])

# Split data
X = data[['Age', 'Gender', 'Education Level', 'Marital Status', 
          'Tenure', 'Job Role', 'Department', 'Salary', 'Work Location']]
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train model
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Test model
y_pred = dt.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.3f}")

# Show top features
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 3 Features:")
print(importance.head(3))

# Show tree
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=X.columns.tolist(), 
          class_names=['Stayed', 'Left'], filled=True)
plt.show()
```

## Output:
![decision tree classifier model](sam.png)
<img width="1510" height="753" alt="Screenshot 2025-10-07 114629" src="https://github.com/user-attachments/assets/0d2d75c2-3be7-4103-a72b-a31f1c43740e" />
<img width="1343" height="723" alt="Screenshot 2025-10-07 114723" src="https://github.com/user-attachments/assets/63116183-ab3c-4b45-aef8-f1cf0c935c07" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

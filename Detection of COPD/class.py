import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("/Users/knithin/Desktop/Capstone_2020_23150_2/copd_data_project.csv")

# Checking missing values
print(dataset.isnull().sum())

# Fill missing values in 'FEV1_phase2' with mean
mean_fev1 = dataset['FEV1_phase2'].mean()
dataset['FEV1_phase2'].fillna(mean_fev1, inplace=True)

# Encoding categorical columns
label_encoder = LabelEncoder()
for column in dataset.select_dtypes(include=['object']).columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Correlation matrix visualization
plt.figure(figsize=(20, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Check correlation with target variable (COPD)
numerical_columns = dataset.select_dtypes(include=['int32', 'int64', 'float64']).columns
correlation_with_copd = dataset[numerical_columns].corrwith(dataset['copd']).abs().sort_values(ascending=False)
print("Correlation with COPD (Target Variable):")
print(correlation_with_copd)

# Statistics of FEV1
print("Max Value of FEV1:", dataset['FEV1'].max())
print("Min Value of FEV1:", dataset['FEV1'].min())

# Display first few rows of the dataset and its info
print(dataset.head())
print(dataset.info())
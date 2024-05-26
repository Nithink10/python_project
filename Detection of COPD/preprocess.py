
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Loading the Data
dataset = pd.read_csv("/Users/knithin/Desktop/Capstone_2020_23150_2/copd_data_project.csv")

# Analyzing The dataset
dataset.info()

# Finding Null Values
dataset.isnull().sum() 

# Handling Null Values
# Calculate the mean of non-null values
mean_fev1 = dataset['FEV1_phase2'].mean()
# Fill null values with the mean                       
dataset['FEV1_phase2'].fillna(mean_fev1, inplace=True)
dataset.isnull().sum()

# Label Encoding
categorical_columns = ['gender', 'asthma', 'bronchitis_attack', 
                       'pneumonia', 'chronic_bronchitis', 'emphysema', 'copd', 'sleep_apnea', 
                       'smoking_status']
# Create a LabelEncoder object
label_encoder = LabelEncoder()
# Iterate through each column in your dataset that contains categorical variables
for column in categorical_columns:
    # Fit label encoder and transform values
    dataset[column] = label_encoder.fit_transform(dataset[column])
    # Print the mapping of original values to numerical labels
    print(f"Column: {column}")
    print("Original Value -> Encoded Label:")
    for original_value, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{original_value} -> {encoded_label}")
    print()

# Correlation Analysis
# Select only the numerical columns
numerical_columns = dataset.select_dtypes(include=['int32', 'int64', 'float64'])
# Calculate the correlation between numerical columns and the target variable (copd)
correlation_with_copd = numerical_columns.corrwith(dataset['copd'])
# Sort the correlations in descending order to identify the most informative columns
correlation_with_copd = correlation_with_copd.abs().sort_values(ascending=False)
# Print the correlation values for each numerical column with the target variable (copd)
print("Correlation with COPD (Target Variable):")
print(correlation_with_copd)

# Print the max and min values of the 'FEV1' column
print("Max Value of FEV1:", max(dataset['FEV1']))
print("Min Value of FEV1:", min(dataset['FEV1']))

# Now you can proceed with the code you provided earlier
# Define the bin intervals based on FEV1 values
bin_edges = [-1.0, 1.0, 1.5, 2.5, 3.0, 5.0, 5.26]
# Define the labels for the bins
bin_labels = ['No COPD', 'Very Low', 'Mild',  'Moderate', 'Severe' , 'Very Severe']
# Create a new column 'Output' to store the bin labels
dataset['Output'] = pd.cut(dataset['FEV1'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Label Encode output column
categorical_columns = ['Output']
# Create a LabelEncoder object
label_encoder = LabelEncoder()
# Iterate through each column in your dataset that contains categorical variables
for column in categorical_columns:
    # Fit label encoder and transform values
    dataset[column] = label_encoder.fit_transform(dataset[column])
    # Print the mapping of original values to numerical labels
    print(f"Column: {column}")
    print("Original Value -> Encoded Label:")
    for original_value, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{original_value} -> {encoded_label}")
    print()

# Assuming these are your input features
input_features = ['visit_age', 'gender', 'height_cm', 'weight_kg', 'sysBP', 'diasBP',
       'hr', 'O2_hours_day', 'bmi', 'asthma', 'bronchitis_attack', 'pneumonia',
       'chronic_bronchitis', 'emphysema', 'sleep_apnea',
       'SmokStartAge', 'CigPerDaySmokAvg', 'Duration_Smoking',
       'smoking_status', 'total_lung_capacity', 'pct_emphysema',
       'functional_residual_capacity', 'pct_gastrapping', 'insp_meanatt',
       'exp_meanatt', 'FEV1_FVC_ratio', 'FEV1', 'FVC', 'FEV1_phase2']

# Store input features in X
X = dataset[input_features]
# Store output features in y
y = dataset['Output']

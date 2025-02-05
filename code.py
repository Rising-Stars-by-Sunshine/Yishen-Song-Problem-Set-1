#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load the dataset
train_file = os.path.join( "adult.data")
test_file = os.path.join( "adult.test")

# Define column names based on UCI repository
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Read the dataset
train_df = pd.read_csv(train_file, names=columns, sep=',', skipinitialspace=True)
test_df = pd.read_csv(test_file, names=columns, sep=',', skipinitialspace=True, skiprows=1)

# Combine datasets for uniform preprocessing
df = pd.concat([train_df, test_df], ignore_index=True)

# Display basic dataset information
print("Dataset Info:")
df.info()
print("\nFirst five rows:")
display(df.head())

# Identify missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
display(missing_values[missing_values > 0])

# Summary statistics
print("\nSummary Statistics:")
display(df.describe())

# Visualizing distributions
plt.figure(figsize=(12, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Count plot for income distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='income', data=df)
plt.title("Income Distribution")
plt.show()

# Checking categorical feature distributions
plt.figure(figsize=(12, 6))
sns.countplot(y=df['education'], order=df['education'].value_counts().index)
plt.title("Education Level Distribution")
plt.show()

# Checking for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Save the processed dataset for further modeling
df.to_csv(os.path.join( "processed_adult.csv"), index=False)

print("\nPreprocessing completed. Processed dataset saved in data folder.")


# In[ ]:





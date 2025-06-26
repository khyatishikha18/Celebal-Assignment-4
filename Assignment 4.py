import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn (you can also load a CSV file if needed)
df = sns.load_dataset('titanic')

# Show first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset Info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.show()

# Univariate Analysis: Histograms for numerical data
df.select_dtypes(include=['float64', 'int64']).hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# Box Plots: Detect outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['age', 'fare']])
plt.title("Box Plot of Age and Fare")
plt.show()

# Correlation Matrix (for numeric columns)
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap of Correlations
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Bivariate Analysis - Survival rate vs class
plt.figure(figsize=(6,4))
sns.barplot(x='class', y='survived', data=df)
plt.title("Survival Rate by Class")
plt.show()

# Pairplot (optional - for numeric interactions)
sns.pairplot(df.dropna(subset=['age', 'fare']), vars=['age', 'fare'], hue='survived')
plt.suptitle("Pairplot of Age and Fare by Survival")
plt.show()

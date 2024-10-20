import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Create necessary directories if they don't exist
os.makedirs('../logs', exist_ok=True)
os.makedirs('../output', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='../logs/eda.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper function to save plots
def save_plot(fig, name):
    output_path = f'../output/{name}.png'
    fig.savefig(output_path, bbox_inches='tight')
    logging.info(f"Saved plot: {output_path}")

# Load datasets
try:
    fraud_data = pd.read_csv('./data/Fraud_Data.csv')
    ip_data = pd.read_csv('./data/IpAddress_to_Country.csv')
    credit_data = pd.read_csv('./data/creditcard.csv')
    logging.info("Successfully loaded datasets")
except FileNotFoundError as e:
    logging.error(f"Error loading datasets: {e}")
    raise

# 1. Summary Statistics
def summarize_data(data, name):
    summary = data.describe().transpose()
    logging.info(f"Summary of {name}:\n{summary}")

summarize_data(fraud_data, 'Fraud Data')
summarize_data(credit_data, 'Credit Card Data')

# 2. Check for Duplicates
def check_duplicates(data, name):
    duplicates = data.duplicated().sum()
    logging.info(f"{name} - Duplicates: {duplicates}")

check_duplicates(fraud_data, 'Fraud Data')
check_duplicates(credit_data, 'Credit Card Data')

# 3. Check Class Imbalance
def plot_class_distribution(data, class_column, name):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=class_column, data=data)
    plt.title(f'{name} - Class Distribution')
    save_plot(plt, f'{name}_class_distribution')
    plt.close()  # Close the plot to release memory

plot_class_distribution(fraud_data, 'class', 'Fraud Data')
plot_class_distribution(credit_data, 'Class', 'Credit Card Data')

# 4. Correlation Heatmap
def plot_correlation_heatmap(data, name):
    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_data.empty:
        logging.warning(f"No numeric data available in {name} for correlation.")
        return

    corr = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'{name} - Correlation Heatmap')
    save_plot(plt, f'{name}_correlation_heatmap')
    plt.close()  # Close the plot to release memory


# 5. Outlier Detection using Boxplot
def plot_outliers(data, feature, name):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'{name} - {feature} Outliers')
    save_plot(plt, f'{name}_{feature}_outliers')
    plt.close()  # Close the plot to release memory

plot_outliers(fraud_data, 'purchase_value', 'Fraud Data')
plot_outliers(credit_data, 'Amount', 'Credit Card Data')

logging.info("EDA Completed.")

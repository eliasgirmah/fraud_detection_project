import pandas as pd
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('../logs', exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    filename='../logs/data_loader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """Load a CSV file and log the process."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {str(e)}")
        return None

# Load all datasets
fraud_data = load_data('./data/Fraud_Data.csv')
ip_data = load_data('./data/IpAddress_to_Country.csv')
credit_data = load_data('./data/creditcard.csv')

# Check for missing values and log them
def log_missing_values(data, dataset_name):
    """Log missing values in the dataset."""
    missing_values = data.isnull().sum()
    logging.info(f"Missing values in {dataset_name}:\n{missing_values}")

if fraud_data is not None:
    log_missing_values(fraud_data, "Fraud_Data")
if ip_data is not None:
    log_missing_values(ip_data, "IpAddress_to_Country")
if credit_data is not None:
    log_missing_values(credit_data, "CreditCard Data")

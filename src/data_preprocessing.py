import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(filename='./logs/preprocessing.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data():
    try:
        # Load the fraud data
        fraud_data = pd.read_csv('./data/Fraud_Data.csv')
        
        # Check for missing values and log the info
        missing_values = fraud_data.isnull().sum()
        logging.info(f"Missing values per column:\n{missing_values}")

        # Remove duplicates
        fraud_data_cleaned = fraud_data.drop_duplicates()
        logging.info(f"Original size: {fraud_data.shape}, Cleaned size: {fraud_data_cleaned.shape}")

        # Save the cleaned data
        output_file = os.path.abspath('./data/fraud_data_cleaned.csv')
        fraud_data_cleaned.to_csv(output_file, index=False)
        logging.info(f"Data cleaned and saved to {output_file}")

    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")

if __name__ == '__main__':
    load_and_clean_data()

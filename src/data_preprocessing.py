import os
import pandas as pd
import logging
from datetime import datetime

# Ensure the log directory exists
log_dir = '../log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, 'preprocessing.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data():
    try:
        # Define input and output paths
        input_file = './data/fraud_data.csv'
        output_file = './data/fraud_data_featured.csv'
        ip_data_file = './data/IpAddress_to_Country.csv'

        # Check if the data directory exists
        if not os.path.exists('./data'):
            os.makedirs('./data')
            logging.warning('Data directory was not found, created it at ./data.')

        # Check if the input file exists
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load the fraud data
        fraud_data = pd.read_csv(input_file)
        logging.info("Data loaded successfully.")

        # Time-Based Features
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

        fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
        fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
        fraud_data['time_difference'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600  # in hours

        # Merge with IP Address to Country data
        if os.path.exists(ip_data_file):
            ip_data = pd.read_csv(ip_data_file)
            ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].apply(lambda x: int(x))
            ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].apply(lambda x: int(x))
            
            # Convert IP addresses in fraud_data to integers
            fraud_data['ip_address'] = fraud_data['ip_address'].apply(lambda x: int(x))

            # Merge the two datasets on the IP address range
            fraud_data = pd.merge(fraud_data, ip_data, 
                                  how='left',
                                  left_on='ip_address',
                                  right_on='lower_bound_ip_address')
            fraud_data.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
            logging.info("Merged fraud data with IP address country data.")

        # Transaction Frequency (count of transactions by the same user)
        fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')

        # Remove duplicates
        fraud_data_cleaned = fraud_data.drop_duplicates()
        logging.info(f"Original size: {fraud_data.shape}, Cleaned size: {fraud_data_cleaned.shape}")

        # Save the cleaned and featured data
        fraud_data_cleaned.to_csv(output_file, index=False)
        logging.info(f"Data with features saved to {os.path.abspath(output_file)}")

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")

if __name__ == '__main__':
    load_and_clean_data()

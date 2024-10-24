import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(filename='./logs/preprocessing.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def feature_engineering(df):
    try:
        # Create 'transaction_hour' and 'transaction_day_of_week' features
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['transaction_hour'] = df['purchase_time'].dt.hour
        df['transaction_day_of_week'] = df['purchase_time'].dt.dayofweek

        # Transaction frequency based on user_id
        df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')

        # One-hot encoding for categorical variables (e.g., 'browser', 'source')
        df = pd.get_dummies(df, columns=['browser', 'source'], drop_first=True)

        logging.info("Feature engineering completed.")
        return df

    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        raise

def normalize_and_scale(df):
    try:
        # Normalize 'purchase_value' using StandardScaler
        scaler = StandardScaler()
        df['purchase_value_scaled'] = scaler.fit_transform(df[['purchase_value']])

        logging.info("Normalization and scaling completed.")
        return df

    except Exception as e:
        logging.error(f"Error during normalization: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Load the cleaned data
        df_cleaned = pd.read_csv('./data/fraud_data_cleaned.csv')

        # Feature engineering
        df_engineered = feature_engineering(df_cleaned)

        # Normalize and scale features
        df_normalized = normalize_and_scale(df_engineered)

        # Save the final processed data
        df_normalized.to_csv('./data/fraud_data_processed.csv', index=False)
        logging.info("Final dataset with engineered and scaled features saved.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")

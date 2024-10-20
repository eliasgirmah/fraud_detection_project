import pandas as pd

def test_data_cleaning():
    try:
        # Load the cleaned data
        df = pd.read_csv('./data/fraud_data_cleaned.csv')

        # Check for duplicates and missing values
        assert df.duplicated().sum() == 0, "There are still duplicates in the data"
        assert df.isnull().sum().sum() == 0, "There are missing values in the data"
        print("Test passed: No duplicates, no missing values")

    except FileNotFoundError:
        print("Test failed: fraud_data_cleaned.csv not found.")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")

if __name__ == '__main__':
    test_data_cleaning()

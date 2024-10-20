import pandas as pd

def test_feature_engineering():
    try:
        df = pd.read_csv('./data/fraud_data_featured.csv')

        # Test if the feature columns exist
        assert 'hour_of_day' in df.columns
        assert 'day_of_week' in df.columns
        assert 'time_difference' in df.columns
        assert 'transaction_count' in df.columns
        assert 'country' in df.columns

        print("Test passed: All feature columns exist.")
    except AssertionError:
        print("Test failed: Some feature columns are missing.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == '__main__':
    test_feature_engineering()

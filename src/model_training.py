import sys
print(sys.path)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(model, X_train, y_train, X_test, y_test):
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model: {model.__class__.__name__}, Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return accuracy

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Load and prepare the data
        fraud_data = load_data('./data/fraud_data_processed.csv')
        creditcard_data = load_data('./data/creditcard.csv')
        
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_data(fraud_data, 'class')
        X_train_card, X_test_card, y_train_card, y_test_card = prepare_data(creditcard_data, 'Class')

        # Define the models
        models = [
            LogisticRegression(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            MLPClassifier()
        ]

        # Train and evaluate models for fraud data
        for model in models:
            logging.info(f"Training {model.__class__.__name__} on fraud data...")
            train_model(model, X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud)

        # Train and evaluate models for creditcard data
        for model in models:
            logging.info(f"Training {model.__class__.__name__} on credit card data...")
            train_model(model, X_train_card, y_train_card, X_test_card, y_test_card)

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")

import mlflow
import mlflow.sklearn

def train_model_with_mlflow(model, X_train, y_train, X_test, y_test):
    try:
        with mlflow.start_run():
            # Log model name
            mlflow.log_param("model_name", model.__class__.__name__)
            
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Log metrics
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Log model
            mlflow.sklearn.log_model(model, model.__class__.__name__)

            logging.info(f"Model: {model.__class__.__name__}, Accuracy: {accuracy}")
            return accuracy

    except Exception as e:
        logging.error(f"Error during model training with MLflow: {str(e)}")
        raise

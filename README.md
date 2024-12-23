# **Fraud Detection Project**

### **Overview**
This project aims to improve fraud detection for e-commerce and bank transactions using advanced machine learning models. Fraud detection is critical for maintaining financial security, reducing losses, and building trust with customers. This solution utilizes geolocation analysis and transaction patterns to enhance detection.

The project follows the entire pipeline: from data analysis and feature engineering to model building, explainability, and deployment using Flask and Docker. A dashboard will provide insights into fraudulent activities.

### **Project Structure**
```bash
fraud_detection_project/
├── data/                # Raw datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
├── notebooks/           # Jupyter notebooks for EDA and feature engineering
├── src/                 # Python scripts for data processing and model training
├── logs/                # Logs generated by Flask and model training
├── deployment/          # Flask API code, Docker configuration
├── tests/               # Unit tests for data preprocessing and models
├── requirements.txt     # Dependencies for the project
└── README.md            # Overview of the project

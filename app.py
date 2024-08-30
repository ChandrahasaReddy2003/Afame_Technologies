from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Initialize variables
model = None
label_encoders = None

def load_and_preprocess_data():
    global model, label_encoders

    # Load datasets (use your paths or URLs)
    train_df = pd.read_csv('"C:\Users\ecred\Desktop\Afame_Technologies\fraudTrain.csv"')
    test_df = pd.read_csv('"C:\Users\ecred\Desktop\Afame_Technologies\fraudTest.csv"')

    # Combine datasets
    combined_df = pd.concat([train_df, test_df], axis=0)

    # Encode categorical columns
    categorical_columns = ['merchant', 'category', 'first', 'last', 'gender', 'job']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        combined_df[column] = le.fit_transform(combined_df[column])
        label_encoders[column] = le

    # Separate datasets
    train_df = combined_df.iloc[:len(train_df), :]
    test_df = combined_df.iloc[len(train_df):, :]

    # Handle missing values
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
    test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].median())

    categorical_cols = train_df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        most_frequent = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(most_frequent)
        test_df[col] = test_df[col].fillna(most_frequent)

    # Prepare data
    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']

    # Drop non-numeric columns
    non_numeric_columns = ['street', 'city', 'state', 'dob', 'trans_num']
    X_train = X_train.drop(columns=non_numeric_columns, errors='ignore')
    X_test = X_test.drop(columns=non_numeric_columns, errors='ignore')

    # Handle class imbalance
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

def predict(input_data):
    global model, label_encoders

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure input_data has the same feature columns
    X_columns = model.feature_names_in_  # Get the feature names
    input_df = input_df[X_columns]

    # Predict
    prediction = model.predict(input_df)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    prediction = predict(data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    load_and_preprocess_data()
    app.run(debug=True)

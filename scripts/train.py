import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def train_model(X_train, y_train):
    """
    Trains a machine learning model.
    """
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints a report.
    """
    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    return accuracy

def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def run_pipeline(data_path, model_path):
    """
    Executes the full ML pipeline: load, train, evaluate, and save.
    """
    data = load_data(data_path)
    if data is None:
        return

    # Assuming 'target' is the name of the target column
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_model = train_model(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    save_model(trained_model, model_path)

if __name__ == "__main__":
    # Create dummy data for a simple demonstration
    # In a real scenario, you'd load a real dataset
    print("Generating dummy data...")
    dummy_data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    }
    df = pd.DataFrame(dummy_data)

    data_file_path = 'data/dummy_data.csv'
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
    df.to_csv(data_file_path, index=False)

    run_pipeline(data_file_path, 'models/my_model.pkl')

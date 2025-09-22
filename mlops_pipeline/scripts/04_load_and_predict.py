from sklearn.datasets import load_breast_cancer
import mlflow
import sys

def load_and_predict():
    """
    Loads a registered model from MLflow Registry and performs a prediction.
    """
    # NOTE: In a real-world scenario, you should replace 'Staging'
    # with the appropriate alias, e.g., 'Production'.
    MODEL_NAME = "breast-cancer-classifier-prod"
    MODEL_ALIAS = "Staging"

    print(f"Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")

    try:
        # Load the model from the MLflow Registry using its alias.
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        # In case of an error, we exit with a non-zero status code.
        sys.exit(1)
        
    # Load sample data to test the model.
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    sample_data = X[0:1]
    actual_label = y[0]
    
    # Make a prediction using the loaded model.
    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()

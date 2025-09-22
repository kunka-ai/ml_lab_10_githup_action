from sklearn.datasets import load_breast_cancer
import mlflow

def load_and_predict():
    MODEL_NAME = "breast-cancer-classifier-prod"
    MODEL_STAGE = "Staging"

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        return

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    sample_data = X[0:1]
    actual_label = y[0]

    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()

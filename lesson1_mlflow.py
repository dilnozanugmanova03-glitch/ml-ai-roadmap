import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

x, y= make_classification(n_samples=1000, n_features=10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
mlflow.set_experiment("fraud_detection")
with mlflow.start_run(run_name="rf_baseline"):
    n_estimators=200
    max_depth=10

    model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)
    preds=model.predict(x_test)
    acc=accuracy_score(y_test, preds)
    f1=f1_score(y_test, preds)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="FraudDetector")

    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
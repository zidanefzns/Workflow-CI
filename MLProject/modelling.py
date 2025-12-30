import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("PredDeposit_preprocessed.csv")

target_col = "deposit"
X = df.drop(columns=[target_col])
y = df[target_col].fillna(df[target_col].mode()[0])

# ===============================
# Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Handle missing values
# ===============================
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ===============================
# MLflow autolog
# ===============================
mlflow.sklearn.autolog()

# ===============================
# Train model
# ===============================
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features & target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)
print("Model saved as model.pkl")

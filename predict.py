import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Example input (replace with real values)
input_data = [5, 116, 74, 0, 0, 25.6, 0.201, 30]

input_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_array)

if prediction[0] == 1:
    print("Result: The person is Diabetic")
else:
    print("Result: The person is Not Diabetic")

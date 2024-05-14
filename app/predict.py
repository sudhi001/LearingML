import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("diabetes_model.keras")

# Prepare new data for prediction (assuming similar structure)
new_data = pd.DataFrame([
    {
        "age": 32,
        "blood_glucose_level": "240",
        "weight": 75
    },
    {
        "age": 55,
        "blood_glucose_level": "90",
        "weight": 85
    }
])
# Normalize new data
new_data_scaled = StandardScaler().fit_transform(new_data[['age', 'blood_glucose_level', 'weight']])

# Make predictions
predictions = model.predict(new_data_scaled)

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Prediction for sample {i+1}: {'Diabetic' if pred < 0.5 else 'Non-diabetic'}")
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load JSON data
with open('sample.json') as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Normalize blood pressure
df[['systolic_bp', 'diastolic_bp']] = df['observations'].apply(lambda x: pd.Series(str(x['blood_pressure']).split('/')))
df[['systolic_bp', 'diastolic_bp']] = df[['systolic_bp', 'diastolic_bp']].astype(int)

# Create target variable: 1 if blood glucose level is greater than 180, else 0
df['target'] = (df['observations'].apply(lambda x: x['blood_glucose_level']) > 180).astype(int)

# Select features and target
X = df[['age', 'systolic_bp', 'diastolic_bp']]
y = df['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network model
model = Sequential([
    Input(shape=(3,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
model.save("diabetes_model.keras")
# # Convert TensorFlow model to TensorFlow Lite
# Define the representative dataset generator
def representative_dataset_gen():
    for _ in range(100):
        # Here you should provide a sample from your actual dataset
        # For illustration, we'll use random data
        # Yield a batch of input data (in this case, a single sample)
        yield [tf.random.normal((1, 64, 64, 3))]


# Set the TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization to default for int8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define the representative dataset for quantization
converter.representative_dataset = representative_dataset_gen

# Restrict the target spec to int8 for full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Instruct the converter to make the input and output layer as integer
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Convert the model
tflite_model = converter.convert()

# Save the model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
#
# # Save TensorFlow Lite model to file
# with open('diabetes_model.tflite', 'wb') as f:
#     f.write(tflite_model)

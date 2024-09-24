import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Simulating IoT sensor data: temperature, humidity, occupancy, and energy usage
def simulate_iot_data(n_samples=1000):
    """
    Function to simulate IoT sensor data.
    In real life, this data would come from actual IoT sensors.
    """
    timestamp = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    temperature = np.random.uniform(15, 30, n_samples)  # Temperature in degrees Celsius
    humidity = np.random.uniform(20, 80, n_samples)  # Humidity percentage
    occupancy = np.random.randint(0, 2, n_samples)  # Occupancy: 0 = not occupied, 1 = occupied
    energy_usage = 0.5 * temperature + 0.2 * humidity + 2 * occupancy + np.random.normal(0, 0.5,
                                                                                         n_samples)  # Energy usage

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamp,
        'temperature': temperature,
        'humidity': humidity,
        'occupancy': occupancy,
        'energy_usage': energy_usage
    })

    return data


# Function to calculate carbon footprint
def calculate_carbon_footprint(energy_usage):
    """
    Given energy usage in kWh, calculate carbon footprint in kg of CO2.
    Assumes 0.5 kg CO2 per kWh.
    """
    return energy_usage * 0.5


# Generate the simulated data
data = simulate_iot_data(1000)

# Preprocessing: Features (temperature, humidity, occupancy) and Labels (energy usage)
X = data[['temperature', 'humidity', 'occupancy']].values
y = data['energy_usage'].values

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the input data for better neural network performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshaping data for CNN/RNN (CNN expects 3D input, so we add an extra dimension for timesteps)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


### Combined CNN, RNN (LSTM), and FNN (Fully Connected) model ###
def create_combined_model():
    """
    Function to create a neural network combining CNN, RNN (LSTM), and FNN layers.
    CNN captures spatial information, LSTM captures temporal dependencies, and FNN processes the final output.
    """
    model = Sequential()

    # CNN layer: Extracts local patterns (spatial dependencies)
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))

    # RNN (LSTM) layer: Captures temporal dependencies
    model.add(LSTM(50, activation='relu', return_sequences=True))

    # Flatten the output from the previous layers
    model.add(Flatten())

    # FNN (Fully Connected Layers)
    model.add(Dense(128, activation='relu'))  # Fully connected hidden layer
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(Dense(64, activation='relu'))  # Another fully connected hidden layer
    model.add(Dense(1))  # Output layer for energy prediction

    # Compile the model (using Mean Squared Error as a loss for regression task)
    model.compile(optimizer='adam', loss='mse')

    return model


# Instantiate the model
model = create_combined_model()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Make predictions on test data
predictions = model.predict(X_test_scaled)
print(f"Sample Predicted Energy Usage: {predictions[0][0]} kWh")

# Calculate carbon footprint for the predicted energy usage
carbon_footprint = calculate_carbon_footprint(predictions)
print(f"Sample Predicted Carbon Footprint: {carbon_footprint[0][0]} kg CO2")

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot actual vs predicted energy usage
plt.scatter(y_test, predictions)
plt.xlabel('Actual Energy Usage (kWh)')
plt.ylabel('Predicted Energy Usage (kWh)')
plt.title('Actual vs. Predicted Energy Usage')
plt.show()

# Plot actual vs predicted carbon footprint
plt.scatter(calculate_carbon_footprint(y_test), carbon_footprint)
plt.xlabel('Actual Carbon Footprint (kg CO2)')
plt.ylabel('Predicted Carbon Footprint (kg CO2)')
plt.title('Actual vs. Predicted Carbon Footprint')
plt.show()

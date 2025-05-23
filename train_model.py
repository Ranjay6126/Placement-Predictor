# Import pandas for data manipulation
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import RandomForestClassifier for model training
from sklearn.ensemble import RandomForestClassifier

# Import LabelEncoder to convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import joblib to save and load models and encoders
import joblib

# Load the dataset from the CSV file
print("Loading the dataset...")
data = pd.read_csv('Placement_data_full_class.csv')
print(f"Dataset loaded successfully with shape: {data.shape}")

# Drop the 'salary' column since it's not needed for prediction
data = data.drop('salary', axis=1)
print("Dropped 'salary' column as it's not needed for prediction")

# Print the number of missing values in each column before handling
print(f"Missing values before handling:\n{data.isnull().sum()}")

# Fill missing values with the mean of each column
data = data.fillna(data.mean())
print("Handled missing values")

# Display unique values in the target column 'status'
print(f"Unique values in 'status' column: {data['status'].unique()}")

# Separate the dataset into features (X) and target (y)
X = data.drop('status', axis=1)  # Features
y = data['status']               # Target variable (Placed/Not Placed)
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Prepare to encode categorical variables
print("Encoding categorical variables...")
encoders = {}  # Dictionary to store label encoders for each categorical column

# List of columns that contain categorical data
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

# Encode each categorical column using LabelEncoder
for col in categorical_cols:
    print(f"Encoding column: {col} with unique values: {X[col].unique()}")
    le = LabelEncoder()               # Create a LabelEncoder instance
    X[col] = le.fit_transform(X[col]) # Encode the column
    encoders[col] = le                # Save the encoder for future use
    print(f"Encoded values: {le.classes_}")

# Save the encoders using joblib for use during prediction
joblib.dump(encoders, 'encoders.pkl')
print("Saved encoders to 'encoders.pkl'")

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Initialize the RandomForestClassifier model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
rf_model.fit(X_train, y_train)

# Evaluate the model on the test data and print the accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Calculate and display feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the trained model to a file using joblib
joblib.dump(rf_model, 'placement_model.pkl')
print("Model trained and saved successfully as 'placement_model.pkl'!")

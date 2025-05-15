import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Generate synthetic data
np.random.seed(42)
data_size = 1000
data = {
    'Time': np.random.randint(0, 24, data_size),
    'Amount': np.random.randint(10, 20000, data_size),
    'Location_Change': np.random.choice([0, 1], data_size, p=[0.9, 0.1]),
    'Device_Change': np.random.choice([0, 1], data_size, p=[0.95, 0.05]),
}

# Step 2: Create label: 1 = fraud, 0 = safe
df = pd.DataFrame(data)
df['Is_Night'] = df['Time'].apply(lambda x: 1 if x < 6 else 0)
df['Fraud'] = df.apply(lambda row: 1 if (row['Amount'] > 10000 or row['Location_Change'] == 1 or row['Device_Change'] == 1 or row['Is_Night'] == 1) else 0, axis=1)

# Step 3: Train AI Model
X = df[['Time', 'Amount', 'Location_Change', 'Device_Change', 'Is_Night']]
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Evaluation
print("=== Model Evaluation ===")
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Real-time AI fraud detection
def get_input():
    print("\n=== Real-time Fraud Check ===")
    time = int(input("Enter time of transaction (0-23): "))
    amount = float(input("Enter amount: "))
    loc_change = int(input("Location changed? (1 = Yes, 0 = No): "))
    device_change = int(input("Device changed? (1 = Yes, 0 = No): "))
    is_night = 1 if time < 6 else 0

    input_data = pd.DataFrame([[time, amount, loc_change, device_change, is_night]],
                              columns=X.columns)

    prediction = model.predict(input_data)[0]
    print("\n➤ Prediction:", "FRAUD ❌" if prediction == 1 else "SAFE ✅")

# Step 6: Loop for user input
while True:
    get_input()
    again = input("\nDo you want to check another transaction? (y/n): ")
    if again.lower() != 'y':
        print("Exiting Fraud Detection AI. Stay safe!")
        break

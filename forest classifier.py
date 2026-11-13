import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------
# 1. LOAD YOUR REAL DATASET FROM THE FOLDER
# ---------------------------------------------------------
# Make sure 'fertilizer.csv' matches the exact name of your file
data = pd.read_csv('fertilizer_recommendation.csv')
print("--- Dataset Loaded Successfully! ---")
print(data.head(3)) # This prints the first 3 rows to verify it loaded

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 2. Encode categorical columns (like Soil_Type, Crop_Type, etc. if present)
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    if col != 'Recommended_Fertilizer':  # Keep target separate if needed
        data[col] = le.fit_transform(data[col])

# 3. Define Features (X) and Target (y)
X = data.drop('Recommended_Fertilizer', axis=1)
y = data['Recommended_Fertilizer']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Check Accuracy
predictions = model.predict(X_test)
print(f"--- Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}% ---")

# 7. Save Model and Columns
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("--- Model Saved Successfully as model.pkl ---")
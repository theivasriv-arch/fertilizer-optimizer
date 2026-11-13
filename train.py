import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset and clean column names
data = pd.read_csv('fertilizer_recommendation.csv')
data.columns = data.columns.str.strip()

# 2. Pick the TRUE text fertilizer target column
text_cols = data.select_dtypes(include=['object', 'string']).columns
fert_cols = [c for c in text_cols if 'fertilizer' in c.lower()]
target_col = fert_cols[0] if fert_cols else text_cols[-1]
print(f"True Target Column Selected: '{target_col}'")

# 3. Drop rows with missing target values and separate X and y
data = data.dropna(subset=[target_col])
X = data.drop(columns=[target_col]).copy()
y = data[target_col].astype(str).copy()

# 4. Process categorical and numerical columns separately and safely
for col in X.columns:
    if X[col].dtype == 'object' or str(X[col].dtype) == 'string':
        # Fill missing text with mode, then encode to numbers
        mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
        X[col] = X[col].fillna(mode_val)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        # Convert to numeric, fill missing numbers with median safely
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"--- Model Accuracy: {accuracy:.2f}% ---")

# 8. Save the trained model and exact feature columns
joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print("--- Model Saved Successfully as model.pkl & model_columns.pkl ---")
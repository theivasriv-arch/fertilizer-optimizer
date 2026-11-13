from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model, columns, and dataset
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')
df = pd.read_csv('fertilizer_recommendation.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the 6 core user inputs from the form
        soil_ph = float(request.form['Soil_pH'])
        nitrogen = float(request.form['Nitrogen_Level'])
        phosphorus = float(request.form['Phosphorus_'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        rainfall = float(request.form['Rainfall'])
        
        # 2. Validation ranges
        if not (3.0 <= soil_ph <= 10.0):
            return render_template('index.html', prediction_text='Error: Soil pH must be between 3.0 and 10.0.')
        
        # 3. Build input data using dataset defaults for remaining columns
        input_data = {}
        for col in model_columns:
            if col in df.select_dtypes(include=['number']).columns:
                input_data[col] = [df[col].median()]
            else:
                le = LabelEncoder()
                encoded_col = le.fit_transform(df[col].astype(str))
                input_data[col] = [encoded_col[0]]
                
        # 4. Override with the user's specific inputs
        input_data['Soil_pH'] = [soil_ph]
        input_data['Nitrogen_Level'] = [nitrogen]
        input_data['Phosphorus_'] = [phosphorus]
        input_data['Potassium'] = [potassium]
        input_data['Temperature'] = [temperature]
        input_data['Rainfall'] = [rainfall]
        
        # Convert to DataFrame matching model expectations
        final_features = pd.DataFrame(input_data, columns=model_columns)
        
        # 5. Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]
        
        return render_template('index.html', prediction_text=f'Recommended Fertilizer: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    # Feature names
    features_name = [
        'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
        'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
        'bland_chromatin', 'normal_nucleoli', 'mitoses'
    ]
    
    # Create DataFrame for prediction
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    # Determine result
    if output[0] == 4:
        res_val = "Breast cancer"
    else:
        res_val = "no Breast cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

# Run the app
if __name__ == "__main__":
    app.run()

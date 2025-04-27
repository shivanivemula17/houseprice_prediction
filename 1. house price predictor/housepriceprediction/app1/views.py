from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from django.conf import settings
from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle
import os

# Load the data once when the server starts
data = pd.read_csv(os.path.join(settings.BASE_DIR, 'static', 'cleaned_data.csv'))

# Define the feature columns
X = data[['location', 'total_sqft', 'bath', 'bhk']]  # Example of feature columns

# Create a ColumnTransformer with OneHotEncoder for the 'location' column
encoder = ColumnTransformer(
    transformers=[
        ('location', OneHotEncoder(drop='first'), ['location']),  # Handle categorical column
    ], 
    remainder='passthrough'  # Keep other columns as is (total_sqft, bath, bhk)
)

# Initialize the machine learning model (e.g., Ridge regression)
model = Ridge()

# Create a pipeline with both preprocessing (encoder) and the model
pipeline = Pipeline([
    ('encoder', encoder),  # First step: Apply OneHotEncoder
    ('model', model),  # Second step: Apply the trained regression model
])

# Train your model using your data (assuming you have 'X' and 'y' ready)
X_train = X[['location', 'total_sqft', 'bath', 'bhk']]
y_train = data['price']  # Assuming your target column is 'price'
pipeline.fit(X_train, y_train)

# Save the trained model to a pickle file
model_path = os.path.join(settings.BASE_DIR, 'static', 'RidgeModel.pkl')
pickle.dump(pipeline, open(model_path, 'wb'))


def index(request):
    # Sort the unique locations
    location = sorted(data['location'].unique())
    return render(request, 'index.html', {'location': location})

def predict(request):
    # Use POST for form submission
    location = request.POST['location']
    bhk = float(request.POST['bhk'])  # Convert to float for prediction
    bath = float(request.POST['bath'])  # Convert to float for prediction
    sqft = float(request.POST['total_sqft'])  # Convert to float for prediction
    
    # Prepare input data for prediction
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Load the saved model pipeline
    model_path = os.path.join(settings.BASE_DIR, 'static', 'RidgeModel.pkl')
    pipeline = pickle.load(open(model_path, 'rb'))

    # Make the prediction
    prediction = pipeline.predict(input_data)[0] * 100000

    return str(np.round(prediction, 2))

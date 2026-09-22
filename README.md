# Placement Prediction System

A Flask web application that estimates a student's placement outcome from academic and profile information. It uses a trained Random Forest classifier and provides an authenticated prediction workflow, follow-up recommendations, and a placement-preparation chatbot.

## Features

- Account signup, login, and logout with Flask-Login.
- Placement predictions from 13 academic and profile inputs.
- Result explanation, confidence score, feature-importance view, and tailored recommendations.
- Responsive interface for desktop and mobile.
- Placement Assistant chatbot for resume, interview, and career-preparation guidance. It works with a built-in guidance mode by default and can use Gemini when an API key is configured.

## Run locally

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   powershell
   pip install -r requirements.txt


3. Start the application:

   powershell
   python app.py


4. Open http://127.0.0.1:5000/ in a browser.

## Optional Gemini chatbot setup

The chatbot works without setup using placement-preparation guidance. To enable Gemini responses, provide a key before starting the app:

powershell
$env:GEMINI_API_KEY = "your-api-key"
python app.py

## Model and data

The model uses these inputs: gender; secondary and higher-secondary percentages and boards; higher-secondary stream; degree type and percentage; work experience; employability-test percentage; MBA specialization and percentage.

To retrain the model after updating the dataset, run:

powershell
python scripts/train_model.py

The script reads data/Placement_data_full_class.csv` and writes the updated model files into `models/.

## Important note

Predictions are estimates based on historical data. They are intended to help students focus their placement preparation and do not guarantee a placement outcome.

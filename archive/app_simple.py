from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify  # Import necessary Flask modules and utilities
import pandas as pd  # For handling data in tabular form
import numpy as np  # For numerical operations
import joblib  # For loading ML models and encoders
import os  # For operating system level operations
from functools import wraps  # For creating decorators like login_required
from sklearn.ensemble import RandomForestClassifier  # Random forest model (used in model training)
import json  # For working with JSON data
import google.generativeai as genai  # For using Google Generative AI API

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Secret key for session management

# Configure Gemini API
genai.configure(api_key="AIzaSyDkkeFqxggpxHebLRPmPhu5PO_3OeDUXGg")  # Setup Gemini API key

# Simple user database (in-memory for demonstration)
users = {}  # Dictionary to store user data temporarily

# Simple authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:  # Check if user is logged in
            flash('Please log in to access this page.')  # Show login required message
            return redirect(url_for('login'))  # Redirect to login page
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')  # Render home page

@app.route('/about')
def about():
    return render_template('about.html')  # Render about page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if username already exists
        if username in users:
            flash('Username already exists!')
            return redirect(url_for('signup'))

        # Create new user (password should be hashed in real apps)
        users[username] = {
            'email': email,
            'password': password
        }

        flash('Account created successfully!')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if user exists and password is correct
        if username in users and users[username]['password'] == password:
            session['username'] = username  # Store user in session
            return redirect(url_for('predict'))
        else:
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove user from session
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Load the model and encoders
            model = joblib.load('placement_model.pkl')
            encoders = joblib.load('encoders.pkl')

            # Get form data
            sl_no = int(request.form.get('sl_no'))
            gender = request.form.get('gender')
            ssc_p = float(request.form.get('ssc_p'))
            ssc_b = request.form.get('ssc_b')
            hsc_p = float(request.form.get('hsc_p'))
            hsc_b = request.form.get('hsc_b')
            hsc_s = request.form.get('hsc_s')
            degree_p = float(request.form.get('degree_p'))
            degree_t = request.form.get('degree_t')
            workex = request.form.get('workex')
            etest_p = float(request.form.get('etest_p'))
            specialisation = request.form.get('specialisation')
            mba_p = float(request.form.get('mba_p'))

            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'sl_no': [sl_no],
                'gender': [gender],
                'ssc_p': [ssc_p],
                'ssc_b': [ssc_b],
                'hsc_p': [hsc_p],
                'hsc_b': [hsc_b],
                'hsc_s': [hsc_s],
                'degree_p': [degree_p],
                'degree_t': [degree_t],
                'workex': [workex],
                'etest_p': [etest_p],
                'specialisation': [specialisation],
                'mba_p': [mba_p]
            })

            # Apply the same encodings as during training
            categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
            for col in categorical_cols:
                if col in encoders:
                    input_data[col] = encoders[col].transform(input_data[col])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Get prediction probability
            prediction_proba = model.predict_proba(input_data)[0]
            confidence = prediction_proba[1] if prediction == 'Placed' else prediction_proba[0]
            confidence_percentage = round(confidence * 100, 2)

            # Store prediction data in session for recommendation page
            session['prediction_data'] = {
                'prediction': prediction,
                'confidence': confidence_percentage,
                'input_data': {
                    'gender': gender,
                    'ssc_p': ssc_p,
                    'ssc_b': ssc_b,
                    'hsc_p': hsc_p,
                    'hsc_b': hsc_b,
                    'hsc_s': hsc_s,
                    'degree_p': degree_p,
                    'degree_t': degree_t,
                    'workex': workex,
                    'etest_p': etest_p,
                    'specialisation': specialisation,
                    'mba_p': mba_p
                }
            }

            return render_template('result.html', prediction=prediction, confidence=confidence_percentage)

        except Exception as e:
            flash(f'Error in prediction: {str(e)}')
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/recommendation')
@login_required
def recommendation():
    if 'prediction_data' not in session:
        flash('Please make a prediction first.')
        return redirect(url_for('predict'))

    prediction_data = session['prediction_data']
    prediction = prediction_data['prediction']

    # Get feature importances from the model
    try:
        model = joblib.load('placement_model.pkl')
        feature_importances = model.feature_importances_

        # Map feature importances to feature names
        feature_names = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 
                        'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
        importance_dict = dict(zip(feature_names, feature_importances))

        # Sort features by importance
        sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Prepare feature importance data for the template
        feature_importance_data = []
        for feature, importance in sorted_importances[:5]:
            readable_name = {
                'ssc_p': 'Secondary Education %',
                'hsc_p': 'Higher Secondary %',
                'degree_p': 'Degree Percentage',
                'mba_p': 'MBA Percentage',
                'etest_p': 'Employability Test %',
                'workex': 'Work Experience',
                'specialisation': 'MBA Specialization',
                'degree_t': 'Degree Type',
                'gender': 'Gender',
                'hsc_s': 'Higher Secondary Stream',
                'ssc_b': 'Secondary Board',
                'hsc_b': 'Higher Secondary Board'
            }.get(feature, feature)

            feature_importance_data.append({
                'feature': readable_name,
                'importance': round(importance * 100, 1)
            })
    except Exception as e:
        print(f"Error getting feature importances: {str(e)}")
        feature_importance_data = []

    input_data = prediction_data.get('input_data', {})

    return render_template('recommendation.html', 
                           prediction=prediction, 
                           feature_importances=feature_importance_data,
                           input_data=input_data)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json.get('message', '')  # Get user message from frontend

        # Set configuration for the generative model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
        }

        # Initialize generative model with configuration
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config
        )

        # Define the chatbot's context and behavior
        context = """You are a placement assistant chatbot for a college placement portal. 
        Your role is to help students with their placement-related queries. 
        Provide information about interview preparation, resume building, placement processes, 
        and career guidance. Be concise, helpful, and encouraging.
        Only answer questions related to placements and career guidance.
        For any other queries, politely redirect the conversation to placement-related topics."""

        # Start a new chat and generate response
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{context}\n\nUser query: {user_message}")

        return jsonify({'response': response.text})  # Send response back to client
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Initialize Flask app
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'placement_model.pkl'
ENCODERS_PATH = MODEL_DIR / 'encoders.pkl'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'placement-predictor-development-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable tracking to save resources

# Gemini is optional. Set GEMINI_API_KEY in the environment to enable it.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize database with SQLAlchemy
db = SQLAlchemy(app)

# Initialize login manager for handling user sessions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to login if not authenticated

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load user from database by ID
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Signup route (GET for form display, POST for form submission)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if username or email already exists
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()

        if user_exists:
            flash('Username already exists!')
            return redirect(url_for('signup'))

        if email_exists:
            flash('Email already exists!')
            return redirect(url_for('signup'))

        # Create a new user with hashed password
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully!')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find user by username
        user = User.query.filter_by(username=username).first()

        # Check credentials
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        # Log the user in
        login_user(user)
        return redirect(url_for('predict'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Prediction route (requires login)
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Load trained model and encoders
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)

            # Get form inputs
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

            # Prepare DataFrame for prediction
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

            # Encode categorical values
            categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
            for col in categorical_cols:
                if col in encoders:
                    input_data[col] = encoders[col].transform(input_data[col])

            # Perform prediction and retain the information needed by recommendations.
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = probabilities[1] if prediction == 'Placed' else probabilities[0]
            session['prediction_data'] = {
                'prediction': prediction,
                'confidence': round(float(confidence) * 100, 2),
                'input_data': request.form.to_dict()
            }

            return render_template('result.html', prediction=prediction, confidence=round(float(confidence) * 100, 2))

        except Exception as e:
            flash(f'Error in prediction: {str(e)}')
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/recommendation')
@login_required
def recommendation():
    prediction_data = session.get('prediction_data')
    if not prediction_data:
        flash('Please make a prediction first.')
        return redirect(url_for('predict'))

    model = joblib.load(MODEL_PATH)
    feature_names = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
    labels = {
        'ssc_p': 'Secondary Education %', 'hsc_p': 'Higher Secondary %',
        'degree_p': 'Degree Percentage', 'mba_p': 'MBA Percentage',
        'etest_p': 'Employability Test %', 'workex': 'Work Experience',
        'specialisation': 'MBA Specialization'
    }
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True)[:5]
    feature_importances = [{'feature': labels.get(name, name.replace('_', ' ').title()), 'importance': round(float(value) * 100, 1)} for name, value in ranked]
    return render_template('recommendation.html', prediction=prediction_data['prediction'], feature_importances=feature_importances, input_data=prediction_data['input_data'])

def local_chat_response(message):
    """Helpful fallback when no Gemini key is configured or its service is unavailable."""
    text = message.lower()
    if any(word in text for word in ('resume', 'cv')):
        return 'Keep your resume to one page: lead with skills and projects, quantify outcomes, and tailor it to the job description.'
    if any(word in text for word in ('interview', 'hr round', 'technical')):
        return 'Prepare a 60-second introduction, use STAR examples for behavioral questions, and practise explaining two projects clearly.'
    if any(word in text for word in ('placement', 'job', 'prepare', 'career')):
        return 'Create a weekly plan: strengthen one technical skill, complete one project improvement, practise aptitude, and apply to relevant roles.'
    return 'I can help with placement preparation, resumes, interviews, aptitude, projects, and career planning. What would you like to work on?'

# Create tables during application setup (compatible with current Flask versions).
with app.app_context():
    db.create_all()

# Chatbot route using Gemini API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = (request.get_json(silent=True) or {}).get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Please enter a message.'}), 400
    try:
        if not (genai and GEMINI_API_KEY):
            return jsonify({'response': local_chat_response(user_message)})

        model = genai.GenerativeModel('gemini-2.0-flash')

        # Define a context to keep chatbot focused on placement guidance
        context = """You are a placement assistant chatbot for a college placement portal. 
        Your role is to help students with their placement-related queries. 
        Provide information about interview preparation, resume building, placement processes, 
        and career guidance. Be concise, helpful, and encouraging.
        Only answer questions related to placements and career guidance.
        For any other queries, politely redirect the conversation to placement-related topics."""

        # Send message to model
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{context}\n\nUser query: {user_message}")

        return jsonify({'response': response.text})
    except Exception:
        return jsonify({'response': local_chat_response(user_message)})

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True)

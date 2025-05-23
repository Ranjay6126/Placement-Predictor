# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Secret key for sessions and security
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable tracking to save resources

# Configure Gemini API with API key
genai.configure(api_key="AIzaSyDkkeFqxggpxHebLRPmPhu5PO_3OeDUXGg")

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
            model = joblib.load('placement_model.pkl')
            encoders = joblib.load('encoders.pkl')

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

            # Perform prediction
            prediction = model.predict(input_data)[0]

            return render_template('result.html', prediction=prediction)

        except Exception as e:
            flash(f'Error in prediction: {str(e)}')
            return redirect(url_for('predict'))

    return render_template('predict.html')

# Run this before the first request to create tables
@app.before_first_request
def create_tables():
    db.create_all()

# Chatbot route using Gemini API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json.get('message', '')

        # Use Gemini's generative model
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True)

# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash  # Flask core modules
from flask_sqlalchemy import SQLAlchemy  # For database integration
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user  # For user authentication
from werkzeug.security import generate_password_hash, check_password_hash  # For password hashing and verification
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import joblib  # For loading saved models
import os  # For OS-related operations
from datetime import datetime  # For handling date and time

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite DB file path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking for performance

# Initialize database with app
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)  # Connect login manager with Flask app
login_manager.login_view = 'login'  # Redirect to 'login' route if not logged in

# Define User model for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Primary key
    username = db.Column(db.String(100), unique=True)  # Username must be unique
    email = db.Column(db.String(100), unique=True)  # Email must be unique
    password = db.Column(db.String(200))  # Hashed password
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Account creation time

# Load user by ID (used by Flask-Login)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home route
@app.route('/')
def index():
    return render_template('index.html')  # Show home page

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')  # Get username from form
        email = request.form.get('email')  # Get email from form
        password = request.form.get('password')  # Get password from form

        # Check if username or email already exists
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()

        if user_exists:
            flash('Username already exists!')  # Notify user
            return redirect(url_for('signup'))  # Reload signup page

        if email_exists:
            flash('Email already exists!')  # Notify user
            return redirect(url_for('signup'))  # Reload signup page

        # Create new user with hashed password
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()  # Save new user to database

        flash('Account created successfully!')  # Notify user
        return redirect(url_for('login'))  # Redirect to login page

    return render_template('signup.html')  # Show signup form

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')  # Get username
        password = request.form.get('password')  # Get password

        user = User.query.filter_by(username=username).first()  # Find user by username

        # Check if user exists and password is correct
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')  # Error message
            return redirect(url_for('login'))  # Reload login page

        login_user(user)  # Log the user in
        return redirect(url_for('predict'))  # Redirect to prediction page

    return render_template('login.html')  # Show login form

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()  # Log user out
    return redirect(url_for('index'))  # Redirect to home page

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Load saved model and encoders
            model = joblib.load('placement_model.pkl')
            encoders = joblib.load('encoders.pkl')

            # Get user input from form
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

            # Create a DataFrame from form input
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

            # Apply encoders to categorical columns
            categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
            for col in categorical_cols:
                if col in encoders:
                    input_data[col] = encoders[col].transform(input_data[col])

            # Make prediction using model
            prediction = model.predict(input_data)[0]

            return render_template('result.html', prediction=prediction)  # Show result

        except Exception as e:
            flash(f'Error in prediction: {str(e)}')  # Display error
            return redirect(url_for('predict'))  # Reload prediction page

    return render_template('predict.html')  # Show prediction form

# Create database tables before the first request
@app.before_first_request
def create_tables():
    db.create_all()  # Create all tables defined by SQLAlchemy models

# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Start Flask app in debug mode

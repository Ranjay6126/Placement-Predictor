AI-Powered Placement Predictor – Flask & Machine Learning

Placement Predictor is a full-stack web application that uses Machine Learning to predict a student’s placement status based on academic performance and personal attributes. The system is built with a Random Forest Classifier trained on historical placement data and delivered through a simple, responsive web interface.

This project combines Flask for the backend, HTML/CSS/Bootstrap for the frontend, and Scikit-learn for the ML model, making it a practical example of integrating AI into a real web application.

Key Features

Secure user signup and login with Flask-Login

AI-based placement prediction using Random Forest (~77% accuracy)

Automatic preprocessing and encoding of categorical inputs

Clear prediction results with meaningful explanations

Responsive UI that works across devices

Persistent data storage using SQLite and SQLAlchemy

Tech Stack

Backend

Python, Flask

SQLite, SQLAlchemy ORM

Flask-Login for authentication

Machine Learning

Scikit-learn

Random Forest Classifier

Label Encoding & Data Preprocessing

Frontend

HTML, CSS

Bootstrap 5

JavaScript

How the Model Works

The prediction model is trained on a placement dataset using:

Data cleaning and preprocessing

Encoding categorical variables with saved encoders

80/20 train-test split

Model evaluation with ~77% accuracy

Saving the trained model (placement_model.pkl) and encoders (encoders.pkl) for real-time predictions

Input Parameters (13 Features)

The model predicts placement status based on:

Gender

10th and 12th board and percentage

12th stream specialization

Degree type and percentage

Work experience

Employability test percentage

MBA specialization and percentage

And related academic attributes

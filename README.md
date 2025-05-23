# AI-Powered Placement Prediction System

This web application uses machine learning to predict placement status based on academic and personal information. The system is built with a Random Forest Classifier trained on historical placement data.

## Features

- **User Authentication**: Secure signup and login system
- **AI Prediction**: Random Forest model with ~77% accuracy
- **Responsive Design**: Modern UI that works on all devices
- **Data Preprocessing**: Automatic encoding of categorical variables
- **Detailed Results**: Comprehensive prediction results with explanations

## Technical Stack

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Frontend**: HTML, CSS, Bootstrap 5
- **Authentication**: Flask-Login

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the machine learning model:

```bash
python train_model.py
```

4. Run the Flask application:

```bash
python app.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Project Structure

- `app.py`: Main Flask application
- `train_model.py`: Script to train and save the Random Forest model
- `Placement_data_full_class.csv`: Dataset used for training
- `templates/`: HTML templates for the web pages
- `static/`: CSS and other static files
- `encoders.pkl`: Saved label encoders for categorical variables
- `placement_model.pkl`: Trained Random Forest model

## Input Parameters

The prediction is based on 13 parameters:

1. `sl_no`: Serial Number (ID)
2. `gender`: Male / Female
3. `ssc_p`: Secondary Education percentage
4. `ssc_b`: Board of Education for 10th (Central / Others)
5. `hsc_p`: Higher Secondary Education percentage
6. `hsc_b`: Board of Education for 12th (Central / Others)
7. `hsc_s`: Specialization in 12th (Commerce / Science / Arts)
8. `degree_p`: Degree percentage
9. `degree_t`: Type of degree (Sci&Tech / Comm&Mgmt / Others)
10. `workex`: Work experience (Yes / No)
11. `etest_p`: Employability test percentage
12. `specialisation`: MBA specialization (Mkt&Fin / Mkt&HR)
13. `mba_p`: MBA percentage

## Model Training

The Random Forest Classifier is trained on the provided dataset with the following steps:

1. Data preprocessing (handling missing values)
2. Label encoding for categorical variables
3. Training with 80% of the data (20% held for testing)
4. Model evaluation (accuracy: ~77%)
5. Saving the model and encoders for prediction

## License

This project is open-source and available for educational and personal use.

## Acknowledgments

- Dataset: The placement dataset used for training
- Scikit-learn: For the machine learning implementation
- Flask: For the web framework
#   P l a c e m e n t - P r e d i c t o r  
 #   P l a c e m e n t - P r e d i c t o r  
 
# AI_Summative-Repo
# GPA Predictor

## Overview
The GPA Predictor is an advanced web application designed to predict students' GPA based on various features such as academic performance, study habits, personal background, and extracurricular activities. The application leverages machine learning techniques to provide accurate GPA predictions and insights into feature importance. The frontend is built using Streamlit, and the backend utilizes a Gradient Boosting model for predictions.

## Features
User Input: The application allows users to input various student-related features.
Prediction: Based on the input features, the model predicts the GPA.
Feature Importance: Displays the importance of various features used in the prediction model.

## Setup Instructions
Prerequisites
Before installation, ensure the following requirements:

-Python 3.7 or higher
-pip (Python package installer)

Installation
To set up the GPA Predictor project on a local machine, follow these steps:

Clone the Repository

First, clone the repository to the local machine using Git:

git clone https://github.com/your-username/gpa_predictor.git
cd gpa_predictor

Create a virtual environment to manage dependencies:
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`


Install the required dependencies using pip:
pip install -r requirements.txt
Run the Application

Run the Streamlit application:
streamlit run gpa_predictor_app.py

Deploying to Streamlit:
Open your web browser and go to http://localhost:8501 to view the application.
Click the deploy button to deploy app.

Deploying to the Cloud
To deploy the application on a cloud platform such as AWS, Azure, or Heroku, follow these general steps:


Deploying on Heroku:
Login to Heroku

Create a new Heroku app:
heroku create gpa-predictor-app

Deploy the app:
git push heroku main

Open the app in your browser

heroku open
Follow similar steps for AWS or Azure, adjusting the commands and settings as necessary for those platforms.

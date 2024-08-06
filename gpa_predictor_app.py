import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
import pickle

st.set_page_config(page_title="GPA Predictor", page_icon="📚", layout="wide")

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        st.success(f"Loaded {file_path} successfully with joblib!")
        return model
    except Exception as e:
        st.warning(f"Couldn't load {file_path} with joblib: {e}")
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            st.success(f"Loaded {file_path} successfully with pickle!")
            return model
        except Exception as e:
            st.error(f"Failed to load {file_path}: {e}")
            return None

feature_importance = {
    'poly_8': 0.24,
    'gradeclass': 0.21,
    'poly_5': 0.17,
    'poly_12': 0.08,
    'grade_improvement': 0.06,
    'g2': 0.05,
    'absences': 0.05,
    'poly_1': 0.03,
    'poly_3': 0.03,
    'studytimeweekly': 0.02
}

df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
df = df.sort_values('Importance', ascending=True)

descriptions = {
    'poly_8': "Complex interaction between multiple factors (including grades and study time)",
    'gradeclass': "Student's current academic standing or grade level",
    'poly_5': "Complex interaction, involving study habits and performance",
    'poly_12': "Interaction between academic and personal factors",
    'grade_improvement': "Progress in grades from one period to the next",
    'g2': "Second to last assesent, indicating recent academic performance",
    'absences': "Number of school absences, reflecting attendance",
    'poly_1': "Interaction involving age and basic academic factors",
    'poly_3': "Interaction of student characteristics and behaviors",
    'studytimeweekly': "Amount of time dedicated to studying each week"
}


df['Description'] = df['Feature'].map(descriptions)

st.header("Feature Importance")

chart = st.bar_chart(df.set_index('Feature')['Importance'])

st.subheader("Feature Descriptions:")
for _, row in df.iterrows():
    st.write(f"**{row['Feature']}**: {row['Description']}")
st.title("GPA Predictor")

model = load_model('gradient_boosting_model.joblib')
scaler = load_model('scaler.joblib')
feature_selector = load_model('feature_selector.joblib')

if model is None or scaler is None or feature_selector is None:
    st.error("Failed to load necessary components. Please check your files.")
    st.stop()

expected_features = [
    'poly_0', 'poly_1', 'poly_2', 'poly_3', 'poly_4', 'poly_5', 'poly_6', 'poly_7', 'poly_8', 'poly_9', 'poly_10', 
    'poly_11', 'poly_12', 'poly_13', 'age', 'medu', 'fedu', 'studytime', 'famrel', 'freetime', 'goout', 'dalc', 
    'walc', 'health', 'absences', 'g1', 'g2', 'alcohol_index', 'parent_edu_max', 'grade_improvement', 
    'study_failure_interaction', 'studentid', 'gender', 'ethnicity', 'parentaleducation', 'studytimeweekly', 
    'tutoring', 'parentalsupport', 'extracurricular', 'sports', 'music', 'volunteering', 'gradeclass', 
    'fjob_other', 'reason_home', 'romantic_yes'
]

st.sidebar.header("Enter Student Information")

age = st.sidebar.number_input("Age", min_value=15, max_value=22, value=18)
medu = st.sidebar.selectbox("Mother's Education", options=[0, 1, 2, 3, 4])
fedu = st.sidebar.selectbox("Father's Education", options=[0, 1, 2, 3, 4])
studytime = st.sidebar.number_input("Study Time (hours per week)", min_value=0, max_value=40, value=10)
famrel = st.sidebar.slider("Family Relationship Quality", min_value=1, max_value=5, value=3)
freetime = st.sidebar.slider("Free Time", min_value=1, max_value=5, value=3)
goout = st.sidebar.slider("Going Out", min_value=1, max_value=5, value=3)
dalc = st.sidebar.slider("Workday Alcohol Consumption", min_value=1, max_value=5, value=1)
walc = st.sidebar.slider("Weekend Alcohol Consumption", min_value=1, max_value=5, value=1)
health = st.sidebar.slider("Health Status", min_value=1, max_value=5, value=3)
absences = st.sidebar.number_input("Number of Absences", min_value=0, max_value=100, value=0)
g1 = st.sidebar.number_input("First Period Grade (G1)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
g2 = st.sidebar.number_input("Second Period Grade (G2)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
gender = st.sidebar.selectbox("Gender", options=["M", "F"])
ethnicity = st.sidebar.selectbox("Ethnicity", options=["White", "Black", "Asian", "Hispanic", "Other"])
parentaleducation = st.sidebar.selectbox("Parental Education", options=["Some High School", "High School", "Some College", "College Degree", "Graduate Degree"])
studytimeweekly = st.sidebar.number_input("Weekly Study Time (hours)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
tutoring = st.sidebar.selectbox("Tutoring", options=["Yes", "No"])
parentalsupport = st.sidebar.selectbox("Parental Support", options=["Low", "Medium", "High"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", options=["Yes", "No"])
sports = st.sidebar.selectbox("Sports Participation", options=["Yes", "No"])
music = st.sidebar.selectbox("Music Participation", options=["Yes", "No"])
volunteering = st.sidebar.selectbox("Volunteering", options=["Yes", "No"])
gradeclass = st.sidebar.selectbox("Grade Class", options=[0, 1, 2, 3, 4])
fjob = st.sidebar.selectbox("Father's Job", options=["teacher", "health", "services", "at_home", "other"])
reason = st.sidebar.selectbox("Reason for Choosing School", options=["home", "reputation", "course", "other"])
romantic = st.sidebar.selectbox("In a Romantic Relationship", options=["Yes", "No"])

def create_poly_features(data):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data[['age', 'medu', 'fedu', 'studytime', 'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'absences', 'g1', 'g2']])
    return pd.DataFrame(poly_features, columns=[f'poly_{i}' for i in range(poly_features.shape[1])], index=data.index)

def predict_gpa(input_data):
    try:
        # Calculate derived features
        input_data['alcohol_index'] = (input_data['dalc'] + input_data['walc']) / 2
        input_data['parent_edu_max'] = np.maximum(input_data['medu'], input_data['fedu'])
        input_data['grade_improvement'] = input_data['g2'] - input_data['g1']
        input_data['study_failure_interaction'] = input_data['studytime'] * input_data['failures']

        # Create polynomial features
        numeric_features = ['age', 'medu', 'fedu', 'studytime', 'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'absences', 'g1', 'g2']
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(input_data[numeric_features])
        poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
        input_data = pd.concat([input_data, pd.DataFrame(poly_features, columns=poly_feature_names, index=input_data.index)], axis=1)

        # Create binary features
        input_data['activities_yes'] = 1 if input_data['extracurricular'].iloc[0] == 'Yes' else 0
        input_data['address_U'] = 1  
        input_data['famsize_LE3'] = 1  
        input_data['famsup_yes'] = 1 if input_data['famsup'].iloc[0] == 'yes' else 0
        input_data['fjob_other'] = 1 if input_data['fjob'].iloc[0] == 'other' else 0
        input_data['reason_home'] = 1 if input_data['reason'].iloc[0] == 'home' else 0
        input_data['romantic_yes'] = 1 if input_data['romantic'].iloc[0] == 'Yes' else 0

  
        expected_features = [
            'poly_0', 'poly_1', 'poly_2', 'poly_3', 'poly_4', 'poly_5', 'poly_6', 'poly_7', 'poly_8', 'poly_9', 'poly_10', 
            'poly_11', 'poly_12', 'poly_13', 'age', 'medu', 'fedu', 'studytime', 'famrel', 'freetime', 'goout', 'dalc', 
            'walc', 'health', 'absences', 'g1', 'g2', 'alcohol_index', 'parent_edu_max', 'grade_improvement', 
            'study_failure_interaction', 'studentid', 'gender', 'ethnicity', 'parentaleducation', 'studytimeweekly', 
            'tutoring', 'parentalsupport', 'extracurricular', 'sports', 'music', 'volunteering', 'gradeclass', 
            'fjob_other', 'reason_home', 'romantic_yes', 'activities_yes', 'address_U', 'failures', 'famsize_LE3', 'famsup_yes'
        ]
        
        final_input = pd.DataFrame(index=[0])
        for feature in expected_features:
            if feature in input_data.columns:
                final_input[feature] = input_data[feature]
            else:
                final_input[feature] = 0

       
        input_scaled = scaler.transform(final_input)
        prediction = model.predict(input_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


if st.sidebar.button("Predict GPA", key="predict_gpa"):
    input_data = pd.DataFrame({
        'age': [age],
        'medu': [medu],
        'fedu': [fedu],
        'studytime': [studytime],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'dalc': [dalc],
        'walc': [walc],
        'health': [health],
        'absences': [absences],
        'g1': [g1],
        'g2': [g2],
        'failures': [0], 
        'studentid': [0], 
        'gender': [1 if gender == 'M' else 0],
        'ethnicity': [ethnicity],
        'parentaleducation': [parentaleducation],
        'studytimeweekly': [studytimeweekly],
        'tutoring': [1 if tutoring == 'Yes' else 0],
        'parentalsupport': [1 if parentalsupport == 'Yes' else 0],
        'extracurricular': [1 if extracurricular == 'Yes' else 0],
        'sports': [1 if sports == 'Yes' else 0],
        'music': [1 if music == 'Yes' else 0],
        'volunteering': [1 if volunteering == 'Yes' else 0],
        'gradeclass': [gradeclass],
        'fjob': [fjob],
        'reason': [reason],
        'romantic': [romantic],
        'famsup': ['yes'],  
        'address': ['U'],  
        'famsize': ['LE3'] 
    })
    gpa_prediction = predict_gpa(input_data)
    if gpa_prediction is not None:
        st.success(f"Predicted GPA: {gpa_prediction:.2f}")
        
st.header("About the Model")

st.write("""
This GPA prediction model uses Gradient Boosting to estimate a student's GPA based on various factors.
The model considers academic performance, study habits, family background, and personal characteristics.
Please note that while this model is based on statistical patterns, individual circumstances can vary.
Use this prediction as a general guide rather than a definitive assessment.
""")

st.header("Feature Explanations")

st.write("""
- Polynomial Features (poly_0 to poly_13): These capture complex interactions between numeric variables.
- Academic Factors: Grades (G1, G2), Study Time, Absences, Tutoring
- Personal Factors: Age, Gender, Ethnicity, Romantic Relationship
- Family Factors: Parental Education, Parental Support, Family Relationship
- Lifestyle Factors: Free Time, Going Out, Alcohol Consumption, Health
- School Factors: Reason for Choosing School, Extracurricular Activities, Sports, Music
- Derived Factors: 
  - Alcohol Index: Average of workday and weekend alcohol consumption
  - Parent Education Max: Highest education level between mother and father
  - Grade Improvement: Difference between G2 and G1
  - Study Failure Interaction: Interaction between study time and past failures
""")

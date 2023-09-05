import joblib
import numpy as np
import streamlit as st
import os
from utils import process_new

Model_path = os.path.join(os.getcwd(),'model2', 'xgboost_model.pkl')
##loading the model
model = joblib.load(Model_path)

##stramlit

st.title('House Rent Prediction..')
st.divider()

##input fields
BHK = st.slider('BHK', min_value=1 , max_value=6, step=1)
Size = st.number_input('Size')
Area_Type = st.selectbox('Area type', options=['Super Area','Carpet Area','Built Area'])
City = st.selectbox('City', options=['Mumbai','Chennai','Bangalore', 'Hyderabad', 'Delhi', 'Kolkata'])
Furnishing_Status = st.selectbox('Furnishing Status', options=['Furnished','Semi-Furnished','Unfurnished'])
Tenant_Preferred = st.selectbox('Tenant Preferred', options=['Bachelors','Bachelors/Family','Family'])
Bathroom = st.selectbox('Number of Bathrooms', options=['1', '2', '3', '4', '5', '6', '7', '10'])
Point_of_Contact = st.selectbox('Point_of_Contact', options=['Owner','Agent','Builder'])
day_posted_on = st.number_input('day_posted_on', min_value=1 , max_value=31, step=1)
month_Posted_on = st.number_input('month_Posted_on', min_value=1 , max_value=12, step=1)

if st.button('Predict house rent...'):
    ##concatenate features
    new_data = np.array([BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, 
                         Bathroom, Point_of_Contact, day_posted_on, month_Posted_on])
    ##call the function from utils.py
    X_processed = process_new(X_new=new_data)

    ##model prediction
    y_pred = model.predict(X_processed)[0]


    st.success(f'House rent prediction is : {y_pred}')

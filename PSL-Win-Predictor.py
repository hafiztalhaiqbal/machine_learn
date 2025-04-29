import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from PIL import Image

# Load trained model (No need for XGBoostLabelEncoder in updated code)
xgbc_model = pickle.load(open("./PSL-Win-XGBC-model.pkl", "rb"))

# Title
st.markdown(
    "<h1 style='color:Gold; text-align: center; font-size: 40px;'>Pakistan Super League (PSL) Win Predictor</h1>",
    unsafe_allow_html=True
)

# Display Image
img = Image.open("./PSL-6.jpg")
st.image(img, width=700)

# Sidebar Form
with st.sidebar.form(key='my_form'):
    Team1 = st.selectbox(
        'Select Team Batting First',
        ('Islamabad United', 'Karachi Kings', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
    )

    Team2 = st.selectbox(
        'Select Team Batting Second',
        ('Karachi Kings', 'Islamabad United', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
    )

    target = st.number_input('Target for the Team Batting Second', min_value=1, value=110)
    cur_runs = st.number_input('Current Runs of the Team Batting Second', min_value=0, value=10)
    wickets = st.number_input('Current Wickets of the Team Batting Second', min_value=0, max_value=10, value=2)
    overs = st.number_input('Current Overs Played by the Team Batting Second', min_value=0.0, max_value=20.0, value=5.5, step=0.1)

    submit_button = st.form_submit_button(label='Predict Win %')

# When Form is Submitted
if submit_button:
    # Calculate balls left and runs left
    balls_bowled = int(overs) * 6 + int((overs % 1) * 10)
    balls_left = 120 - balls_bowled
    runs_left = target - cur_runs

    # Prepare input
    input_data = pd.DataFrame({
        "wickets": [wickets],
        "balls_left": [balls_left],
        "runs_left": [runs_left]
    })

    # Make prediction
    prediction = xgbc_model.predict_proba(input_data)[0]

    # Create a pie chart
    fig = px.pie(
        names=[f"{Team1} (Bat First)", f"{Team2} (Bat Second)"],
        values=[prediction[0], prediction[1]],
        title="Match Winning Chances"
    )
    st.plotly_chart(fig)

    # Show interpretation
    st.success(
        f"Interpretation:\n\n"
        f"- {Team1} Win Chance: {round(prediction[0] * 100, 2)}%\n"
        f"- {Team2} Win Chance: {round(prediction[1] * 100, 2)}%"
    )

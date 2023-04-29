import streamlit as sl
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import pickle
import sklearn
#import matplotlib.pyplot as plt

match = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')


sl.sidebar.header("IPL Data Analysis")
sl.sidebar.image('IPL image.png')

options = sl.sidebar.radio("Select an Option", ("Summary","Predictive Analysis"))

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

city = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
       'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
       'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
       'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']


if options == "Summary":
    sl.title("TATA IPL Data Analysis")
    # Bar graph for the team and their number of matches played
    x = match['team1'].value_counts()
    y = match['team2'].value_counts()

    fig = px.bar(x, y, title="Team Wise Winning List")
    sl.plotly_chart(fig)

    sl.header("Year Wise Winning Summary")
    url = pd.read_html('https://www.careerpower.in/ipl-winners-list.html')
    win = url[0].sort_values('Year').drop(index=15)
    sl.table(win)

    sl.header("Most IPL Winner Teams")
    most_ipl = url[3]
    sl.table(most_ipl)

    sl.header("IPL Winning Team Captain, Player of the Series and Man of the Match")
    ipl = url[2]
    sl.table(ipl)


elif options == 'Predictive Analysis':
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    sl.title("TATA IPL Predictive Analysis")
    col1, col2 = sl.columns(2)

    with col1:
        batting_team = sl.selectbox("Select Batting Team: ", sorted(teams))
    with col2:
        bowling_team = sl.selectbox("Select Bowling Team: ", sorted(teams))

    selected_city = sl.selectbox("Select the Host City: ", sorted(city))

    target = sl.number_input('Target')

    col3, col4, col5 = sl.columns(3)

    with col3:
        score = sl.number_input('Score: ')
    with col4:
        overs = sl.number_input('Overs Completed: ')
    with col5:
        wickets = sl.number_input('Wickets: ')

    if sl.button('Predict Winner'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                                 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                                 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr],
                                 'rrr': [rrr]})
        sl.table(input_df)

        result = pipe.predict_proba(input_df)
        # m1 = joblib.load("model")
        # result = m1.predict(input_df)
        loss = result[0][0]
        win = result[0][1]
        sl.header(batting_team + "- " + str(round(win * 100)) + "%")
        sl.header(bowling_team + "- " + str(round(loss * 100)) + "%")

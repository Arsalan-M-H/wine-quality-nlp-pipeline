import joblib
import psycopg2
import pandas as pd
from datetime import date, timedelta

connection = psycopg2.connect(
    host='localhost',
    database='wine_db',
    user='postgres',
    password='password123',
    port='5432')
# Creating connection to database (this is for demonstration purposes, database does not exist)

current_date = date.today()
month = current_date.month
year = current_date.year
week_ago = current_date - timedelta(days=7)
data_query = f""" SELECT * 
    FROM wine_descriptions 
    WHERE date between '{week_ago}' AND '{current_date}'"""
# Creating query to gather review descriptions from the past week, data is dynamic and does not need to be coded in each time


original_data = pd.read_sql(data_query, connection)
data = original_data.copy()
data['description'] = data['description'].fillna('')
data['description'] = data['description'].str.strip()
data['description'] = data['description'].str.lower()
data['description'] = data['description'].str.replace(r'[^a-z\s]', ' ', regex=True)
data['description'] = data['description'].str.replace(r'\s+', ' ', regex=True)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
X = data['description']
# Applying same data processing as when I trained the model, and separating descriptions

data_tfidf = vectorizer.transform(X)
output = model.predict(data_tfidf)
# Running descriptions through our vetorizer and predicting qualities using the model

original_data['quality'] = output
original_data['run_date'] = current_date
original_data.to_csv(f'{year} {month} description_output.csv', index = False)
# Adding quality output and date the model was run to the original dataset, so we can still have access to other features for our dashboard

connection.close()
# Closing the database connection

# app.py
import uuid
import json
import sqlite3
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import joblib
import os
import random
import string
import smtplib


app = Flask(__name__)

# Connect to SQLite DB 
conn = sqlite3.connect('predictions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    screen_time INTEGER,
    sex INTEGER,
    age INTEGER,
    prediction_results TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

# Target variables
target_cols = ['SDQEMOT_C', 'SDQCOND_C', 'SDQHYPE_C', 'SDQPEER_C']
target_names = {
    'SDQEMOT_C': 'Emotional Difficulties',
    'SDQCOND_C': 'Conduct Problems',
    'SDQHYPE_C': 'Hyperactivity',
    'SDQPEER_C': 'Peer Problems'
}

# Load models
models = {}
for target in target_cols:
    model_path = f'models/xgb_model_{target}.pkl'
    if os.path.exists(model_path):
        models[target] = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

def generate_unique_session_id(length=6):
    """Generate a unique 6-letter session ID."""
    while True:
        session_id = ''.join(random.choices(string.ascii_letters, k=6))
        cursor.execute("SELECT 1 FROM predictions WHERE id = ?", (session_id,))
        if not cursor.fetchone():
            return session_id

@app.route('/predict', methods=['POST'])
def predict():
    try:
        screen_time = int(request.form['SCREENTIME_C'])
        sex = int(request.form['SEX_C'])
        age = int(request.form['AGEP_C'])
        

        features = np.array([[screen_time, sex, age]])
        results = {}

        for target in target_cols:
            model = models.get(target)
            if model:
                prediction = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]
                status = 'At Risk' if prediction == 1 else 'Normal'
                friendly_name = target_names.get(target, target)
                results[friendly_name] = {
                    'probability': round(prob * 100, 2),
                    'status': status
                }

        session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))  # shorter ID

        cursor.execute('''INSERT INTO predictions (id, screen_time, sex, age, prediction_results) 
                          VALUES (?, ?, ?, ?, ?)''',
                       (session_id, screen_time, sex, age, json.dumps(results)))
        conn.commit()


        return render_template('dashboard.html', prediction_results=results, session_id=session_id)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

    


@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve():
    if request.method == 'POST':
        session_id = request.form['session_id']
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        if row:
            results = json.loads(row[4])
            return render_template('dashboard.html', prediction_results=results, session_id=session_id)
        else:
            return render_template('retrieve.html', error='ID not found.')
    return render_template('retrieve.html')

@app.route('/update-form', methods=['POST'])  # Changed from '/update'
def show_update_form():
    session_id = request.form['session_id']
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (session_id,))
    record = cursor.fetchone()
    if record:
        return render_template('update.html', record=record)
    else:
        return render_template('retrieve.html', error="No record found for this ID.")


@app.route('/update/<session_id>', methods=['GET', 'POST'])
def update_prediction(session_id):
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return "Session ID not found."

    if request.method == 'POST':
        screen_time = int(request.form['SCREENTIME_C'])
        sex = int(request.form['SEX_C'])
        age = int(request.form['AGEP_C'])
        features = np.array([[screen_time, sex, age]])

        updated_results = {}
        for target in target_cols:
            model = models.get(target)
            if model:
                prediction = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]
                status = 'At Risk' if prediction == 1 else 'Normal'
                friendly_name = target_names.get(target, target)
                updated_results[friendly_name] = {
                    'probability': round(prob * 100, 2),
                    'status': status
                }

        cursor.execute('''UPDATE predictions 
                          SET screen_time=?, sex=?, age=?, prediction_results=? 
                          WHERE id=?''',
                       (screen_time, sex, age, json.dumps(updated_results), session_id))
        conn.commit()

        return render_template('dashboard.html', prediction_results=updated_results, session_id=session_id)

    return render_template('update.html', session_id=session_id, screen_time=row[1], sex=row[2], age=row[3])

if __name__ == '__main__':
    app.run(debug=True)

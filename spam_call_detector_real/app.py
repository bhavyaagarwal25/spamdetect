from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
# When saving
sms_model, word2vec, sms_label_encoder = joblib.load('SPAM1.pkl')


# ✅ Load the call spam detection model
call_model = joblib.load('spam_detection_model.pkl')

MAX_NUMBER_LENGTH = 13
LOG_FILE = 'predictions_log.csv'

# ✅ SMS vectorization using Word2Vec
def get_avg_word2vec(text, model, k=100):
    words = text.lower().split()
    vec = np.zeros(k)
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec.reshape(1, -1)

# Dynamic call time
def get_call_time():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Extract features from input
def extract_features(number, saved, duration, frequency, call_time, spam_reports):
    clean_number = re.sub(r'\D', '', number)
    features = {
        'country_code': 91,
        'starts_with_140': int(clean_number.startswith('140')),
        'repeated_digits': sum(v for v in pd.Series(list(clean_number)).value_counts() if v > 1),
        'previous_spam_reports': int(spam_reports),
        'call_duration': float(duration),
        'call_frequency': float(frequency),
        'call_time': call_time
    }
    return features

# Save to CSV log
def log_prediction(number, prediction, confidence):
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'number': number,
        'prediction': prediction,
        'confidence': confidence
    }
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=entry.keys()).to_csv(LOG_FILE, index=False)
    pd.DataFrame([entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/sms')
def sms_page():
    return render_template('sms.html')

@app.route('/call')
def call_page():
    return render_template('index.html', prediction=None, number="")
@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        vec = get_avg_word2vec(message, word2vec)
        prediction = sms_model.predict(vec)[0]
        label = "🚫 Spam SMS" if prediction == 1 else "✅ Legit SMS"
        return jsonify({'result': label})

    except Exception as e:
        print(f"❌ Error in /predict_sms: {e}")  # ✅ Debug output in terminal
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    number = request.form.get('number', '').strip()
    saved = request.form.get('saved', 'no').strip()
    duration = request.form.get('duration', '0').strip()
    frequency = request.form.get('frequency', '0').strip()
    call_time = request.form.get('time', '').strip()
    spam_reports = request.form.get('spam_reports', '0').strip()

    if not number or not call_time:
        return render_template('index.html', prediction="⚠️ Missing inputs", number=number)

    clean_number = re.sub(r'\D', '', number)
    if not (10 <= len(clean_number) <= MAX_NUMBER_LENGTH):
        return render_template('index.html', prediction="⚠️ Invalid number length", number=number)

    try:
        features = extract_features(clean_number, saved, duration, frequency, call_time, spam_reports)
        input_df = pd.DataFrame([features])
        input_df = pd.get_dummies(input_df)

        expected_columns = [
            'country_code', 'starts_with_140', 'repeated_digits', 'previous_spam_reports',
            'call_duration', 'call_frequency',
            'call_time_Afternoon', 'call_time_Evening', 'call_time_Morning', 'call_time_Night'
        ]

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        if input_df['previous_spam_reports'].iloc[0] > 2:
            prediction = 1
            proba = 1.0
        elif input_df['starts_with_140'].iloc[0] == 1 and input_df['call_duration'].iloc[0] <= 1:
            prediction = 1
            proba = 0.95
        else:
            prediction = call_model.predict(input_df)[0]
            proba = call_model.predict_proba(input_df)[0][1]

        confidence = max(proba, 1 - proba)
        result = "🚫 SPAM" if prediction == 1 else "✅ NOT SPAM"
        message = f"{result} ({confidence:.1%} confidence)"

        log_prediction(clean_number, result, confidence)

        return render_template('index.html', prediction=message, number=number)

    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {str(e)}", number=number)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

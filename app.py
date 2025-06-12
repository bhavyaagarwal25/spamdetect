from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Correctly load the full tuple from SPAM1.pkl
sms_model, word2vec, sms_label_encoder = joblib.load('SPAM1.pkl')

# ✅ Load the call classification model separately
call_model = joblib.load('call_spam_rf_model.pkl')

# ✅ SMS Vectorizer using Word2Vec
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

# ✅ Call Number Vectorizer
def call_vectorizer(number):
    features = [
        len(number),                      # Length of number
        number.startswith("140") * 1,     # Starts with '140'
          # All digits
    ]
    return np.array(features).reshape(1, -1)

# Landing page
@app.route('/')
def landing_page():
    return render_template('landing.html')

# SMS input page
@app.route('/sms')
def sms_page():
    return render_template('sms.html')

# Call input page
@app.route('/call')
def call_page():
    return render_template('call.html')

# ✅ Predict SMS spam or not
@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message'}), 400

    vec = get_avg_word2vec(message, word2vec)
    prediction = sms_model.predict(vec)[0]
    label = "🚫 Spam SMS" if prediction == 1 else "✅ Legit SMS"
    return jsonify({'result': label})

# ✅ Predict Call spam or not
@app.route('/predict_call', methods=['POST'])
def predict_call():
    data = request.get_json()
    number = data.get('number', '')
    if not number:
        return jsonify({'error': 'No number'}), 400

    try:
        vec = call_vectorizer(number)
        prediction = call_model.predict(vec)[0]
        label = "📵 Spam Call" if prediction == 1 else "📞 Safe Call"
        return jsonify({'result': label})
    except Exception as e:
        print("Error in /predict_call:", e)
        return jsonify({'error': 'Internal server error'}), 500
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load vectorizer and model
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text using vectorizer
    vect_text = vectorizer.transform([text])

    # Predict
    pred = model.predict(vect_text)[0]
    result = 'spam' if pred == 1 else 'not_spam'

    return jsonify({"prediction": result})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)

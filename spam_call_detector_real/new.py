import joblib

# Assuming these are defined and trained:
# sms_model         → your SVC model
# word2vec_model    → your Word2Vec model
# label_encoder     → LabelEncoder for 'spam'/'not_spam'

joblib.dump((sms_model, word2vec_model, label_encoder), 'SPAM_PKL.pkl')
print("✅ SPAM_PKL.pkl saved successfully.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv('train_data_with_duration_freq.csv', comment='#')
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df['label'] = df['label'].str.lower().replace({'not_spam': 'not_spam', 'spam': 'spam'})
df['call_duration'] = pd.to_numeric(df['call_duration'], errors='coerce')
df['call_frequency'] = pd.to_numeric(df['call_frequency'], errors='coerce')
df = df.dropna()
df = pd.get_dummies(df, columns=['call_time'], prefix='call_time')

# Encode target
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # spam=1, not_spam=0

# Label distribution
print("Label distribution:\n", df['label'].value_counts())

# Split features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Weight: more for Afternoon + high duration+frequency + spam indicators
sample_weights = []
for idx, row in X_train.iterrows():
    weight = 1
    if row.get('call_time_Afternoon', 0) == 1:
        weight += 1
    if (row['call_duration'] + row['call_frequency']) > 10:
        weight += 1
    if row['starts_with_140'] == 1 or row['previous_spam_reports'] > 1:
        weight += 2
    sample_weights.append(weight)
sample_weights = pd.Series(sample_weights, index=X_train.index)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Feature importance
feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_imp.head())

plt.figure(figsize=(10,5))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance")
plt.show()

# Prediction function with improved threshold
def predict_spam(country_code, starts_with_140, repeated_digits, previous_spam_reports,
                 call_time, call_duration, call_frequency):
    if previous_spam_reports > 2:
        return 'spam', 1.0

    call_time_cols = [col for col in X.columns if col.startswith('call_time_')]
    call_time_features = {col: 0 for col in call_time_cols}
    time_col = f'call_time_{call_time}'
    if time_col in call_time_features:
        call_time_features[time_col] = 1
    else:
        print(f"Warning: unknown call_time '{call_time}'")
        return 'not_spam', 0.0

    input_data = pd.DataFrame({
        'country_code': [country_code],
        'starts_with_140': [starts_with_140],
        'repeated_digits': [repeated_digits],
        'previous_spam_reports': [previous_spam_reports],
        'call_duration': [call_duration],
        'call_frequency': [call_frequency],
        **call_time_features
    })

    input_data = input_data[X.columns]
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba > 0.4 else 0

    print("Input:", input_data)
    print("Spam probability:", proba)

    return label_encoder.inverse_transform([prediction])[0], float(proba)

# Sample predictions
print("\nSample Predictions:")
test_cases = [
    (91, 0, 2, 0, 'Afternoon', 15, 10),
    (91, 0, 1, 0, 'Morning', 2, 1),
    (91, 0, 1, 3, 'Evening', 10, 5),
    (91, 1, 5, 1, 'Night', 1, 1),
]

for case in test_cases:
    result, prob = predict_spam(*case)
    print(f"Input: {case} -> Prediction: {result} (Confidence: {prob:.2%})")

# Save model
joblib.dump(model, 'spam_detection_model.pkl')
print("\nModel saved as 'spam_detection_model.pkl'")

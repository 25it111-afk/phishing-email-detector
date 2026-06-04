import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert labels into numbers
df['label'] = df['label'].map({'safe': 0, 'phishing': 1})

# Features and labels
X = df['text']
y = df['label']

# Convert text into numbers (VERY IMPORTANT STEP)
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Create ML model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Full report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to test new emails
def predict_email(text):
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    return "Phishing" if result[0] == 1 else "Safe"

# Test examples
print("\n--- TEST EXAMPLES ---")
print(predict_email("Your account is locked. Click link to verify"))
print(predict_email("Tomorrow we have class meeting at 10 AM"))
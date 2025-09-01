import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("📥 Loading data...")

# Get base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
data_path = os.path.join(BASE_DIR, "data", "sentiment_data.csv")
if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    exit(1)

# Load dataset
data = pd.read_csv(data_path)
print("✅ Data loaded.")
print("📊 Label distribution:\n", data['label'].value_counts())

# Preprocessing
X = data['text'].astype(str)
y = data['label'].str.lower()

# Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_vec = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Accuracy: {accuracy * 100:.2f}%")
print("📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "sentiment_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("💾 Model and vectorizer saved in 'model/' folder.")
print("✅ Training completed successfully.")

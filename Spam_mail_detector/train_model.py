import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import numpy as np

print("📥 Loading dataset...")

try:
    # ✅ Load dataset from a valid source (tab-separated, no header in file)
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    
    # Display sample data
    print("\n📊 Sample data:")
    print(df.head())
    
    # Check label distribution
    print("\n📈 Label distribution:")
    print(df['label'].value_counts())
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

print("\n🔄 Processing data...")

# ✅ Convert labels to numeric: 1 = ham (not spam), 0 = spam
original_labels = df['label'].copy()
df['label'] = df['label'].map({'ham': 1, 'spam': 0})

# Verify mapping worked correctly
print(f"✅ Label mapping completed:")
print(f"   - Ham (safe emails): {sum(df['label'] == 1)} samples")
print(f"   - Spam emails: {sum(df['label'] == 0)} samples")

# Check for any missing values after mapping
if df['label'].isnull().any():
    print("⚠️ Warning: Found unmapped labels!")
    print(df[df['label'].isnull()]['label'].value_counts())

print("\n🔤 Vectorizing text...")

# ✅ Text vectorization using CountVectorizer with better parameters
vectorizer = CountVectorizer(
    stop_words='english',      # Remove common English stop words
    max_features=5000,         # Limit vocabulary size
    min_df=2,                  # Ignore terms that appear in less than 2 documents
    max_df=0.95,               # Ignore terms that appear in more than 95% of documents
    ngram_range=(1, 2)         # Use both unigrams and bigrams
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

print(f"✅ Vectorization completed! Feature matrix shape: {X.shape}")
print(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")

print("\n🔀 Splitting data...")

# ✅ Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced split
)

print(f"✅ Data split completed!")
print(f"   - Training set: {X_train.shape[0]} samples")
print(f"   - Test set: {X_test.shape[0]} samples")

print("\n🤖 Training model...")

# ✅ Train a Naive Bayes spam classifier
model = MultinomialNB(alpha=1.0)  # Add smoothing parameter
model.fit(X_train, y_train)

print("✅ Model training completed!")

print("\n📊 Evaluating model performance...")

# Make predictions on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed classification report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))

# Confusion matrix
print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("   [[True Spam, False Ham],")
print("    [False Spam, True Ham]]")

print("\n🧪 Testing with sample messages...")

# Test with sample messages
test_messages = [
    "Congratulations! You've won a $1000 gift card! Click here to claim now!",
    "Hey, are you free for lunch tomorrow?",
    "URGENT: Your account will be suspended. Click here immediately!",
    "Meeting rescheduled to 3 PM. See you then.",
    "FREE! Call now to claim your prize! Limited time offer!"
]

for i, message in enumerate(test_messages, 1):
    X_sample = vectorizer.transform([message])
    prediction = model.predict(X_sample)[0]
    probability = model.predict_proba(X_sample)[0]
    
    result = "✅ HAM (Safe)" if prediction == 1 else "⚠️ SPAM"
    confidence = max(probability) * 100
    
    print(f"{i}. Message: '{message[:50]}{'...' if len(message) > 50 else ''}'")
    print(f"   Prediction: {result} (Confidence: {confidence:.1f}%)")
    print()

print("\n💾 Saving model and vectorizer...")

# ✅ Save the trained vectorizer and model as .pickle files
try:
    with open('count_vactor_email.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Vectorizer saved as 'count_vactor_email.pickle'")
    
    with open('spam_email_detactor.pickle', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved as 'spam_email_detactor.pickle'")
    
    # Also save with corrected filenames
    with open('count_vector_email.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Vectorizer also saved as 'count_vector_email.pickle' (corrected name)")
    
    with open('spam_email_detector.pickle', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model also saved as 'spam_email_detector.pickle' (corrected name)")
    
except Exception as e:
    print(f"❌ Error saving files: {e}")
    exit(1)

print("\n🎉 Training completed successfully!")
print("\n📋 Summary:")
print(f"   - Dataset size: {len(df)} messages")
print(f"   - Model accuracy: {accuracy:.4f}")
print(f"   - Label encoding: 0 = SPAM, 1 = HAM (Safe)")
print(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")
print("\n✅ You can now run your GUI application!")

# Save a summary file
summary = {
    'accuracy': accuracy,
    'label_encoding': {'spam': 0, 'ham': 1},
    'dataset_size': len(df),
    'vocabulary_size': len(vectorizer.vocabulary_),
    'test_accuracy': accuracy
}

with open('model_summary.pickle', 'wb') as f:
    pickle.dump(summary, f)
print("📊 Model summary saved as 'model_summary.pickle'")
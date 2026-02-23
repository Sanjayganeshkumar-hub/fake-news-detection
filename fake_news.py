import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")

# Load datasets
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Features and labels
X = data["text"]
y = data["label"]

print("Training model...")

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train on full dataset (since dataset is small)
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Model Trained Successfully!")

# Prediction function
def predict_news(news_text):
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"

# User input loop
while True:
    news = input("\nEnter news text to check (type 'exit' to stop): ")
    if news.lower() == "exit":
        break
    result = predict_news(news)
    print("Prediction:", result)
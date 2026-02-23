import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Preparing model...")

# Load dataset
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

X = data["text"]
y = data["label"]

# NLP processing
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Model ready for real-time prediction!")

# 🔴 ENTER YOUR API KEY HERE
API_KEY = "e3669af214f245faabc32d1164215361"

url = f"https://newsapi.org/v2/everything?q=news&language=en&apiKey={API_KEY}"

response = requests.get(url)
news_data = response.json()

print("\nLive News Predictions:\n")

for article in news_data["articles"]:
    headline = article["title"]

    news_vector = vectorizer.transform([headline])
    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        result = "FAKE"
    else:
        result = "REAL"

    print(f"News: {headline}")
    print(f"Prediction: {result}")
    print("-" * 50)
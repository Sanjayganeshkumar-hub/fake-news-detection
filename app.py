from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests

app = Flask(__name__)

# Load datasets
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

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        news = request.form["news"]
        news_vector = vectorizer.transform([news])
        pred = model.predict(news_vector)

        if pred[0] == 0:
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

    return render_template("index.html", prediction=prediction)

# Live news page
@app.route("/live")
def live():
    API_KEY = "e3669af214f245faabc32d1164215361"
    url = f"https://newsapi.org/v2/everything?q=news&language=en&apiKey={API_KEY}"

    response = requests.get(url)
    news_data = response.json()

    results = []

    for article in news_data["articles"]:
        headline = article["title"]
        news_vector = vectorizer.transform([headline])
        pred = model.predict(news_vector)

        result = "FAKE" if pred[0] == 0 else "REAL"
        results.append((headline, result))

    return render_template("live.html", results=results)

# About page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("models/linear_clf.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_v.pkl")


# Preprocess function
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower()
    review = review.split()
    review = [
        ps.stem(word) for word in review if not word in stopwords.words("english")
    ]
    review = " ".join(review)
    return review


# Define the route for the form
@app.route("/", methods=["GET", "POST"])
def predict_news():
    result = None
    if request.method == "POST":
        user_input = request.form.get("Title")
        user_input = preprocess_text(user_input)
        user_input_vectorized = tfidf_vectorizer.transform([user_input]).toarray()
        prediction = model.predict(user_input_vectorized)
        if prediction[0] == 0:
            result = "FAKE."
        else:
            result = "REAL."
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")

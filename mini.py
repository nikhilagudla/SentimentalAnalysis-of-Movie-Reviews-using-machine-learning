from flask import Flask, request, render_template
import sklearn
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
from nltk import WordNetLemmatizer
import re

wn = WordNetLemmatizer()
model = joblib.load(open("svc_pickle.pkl", "rb"))
tfidf = joblib.load(open('tfidf_pickle.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form['text']
        vector_input = tfidf.transform([text])
        prediction = model.predict(vector_input)
        return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

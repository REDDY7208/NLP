
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load pre-trained model and TF-IDF vectorizer
clf = joblib.load('amazon.pkl')


@app.route('/')
def home():
    return render_template('review.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    review_tfidf = tfidf_vectorizer.transform([review])
    prediction = clf.predict(review_tfidf)[0]
    return f'The sentiment of the review is: {prediction}'

if __name__ == '__main__':
    app.run()

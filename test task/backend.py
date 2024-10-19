from flask import Flask, request, jsonify, render_template
from spamclassifier import SpamMailClassifier  # Assuming your class is in spam_mail_classifier.py

app = Flask(__name__)
classifier = SpamMailClassifier('mail_data.csv')
classifier.train()

# Route to serve the HTML front-end (index.html)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train_model():
    classifier.train()
    return jsonify(classifier.evaluate())

@app.route('/predict', methods=['POST'])
def predict_mail():
    data = request.json
    mail_text = data['mail_text']
    prediction = classifier.predict([mail_text])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

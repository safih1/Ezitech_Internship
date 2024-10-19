import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class SpamMailClassifier:
    def __init__(self, csv_file):
        # Load data
        raw_mail_data = pd.read_csv(csv_file)
        mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
        
        # Label spam as 0, ham as 1
        mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
        mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
        
        # Split data into features and labels
        self.X = mail_data['Message']
        self.Y = mail_data['Category'].astype('int')
        
        # Train-test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=3)
        
        # TF-IDF feature extraction
        self.feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        self.X_train_features = self.feature_extraction.fit_transform(self.X_train)
        self.X_test_features = self.feature_extraction.transform(self.X_test)
        
        # Initialize Logistic Regression model
        self.model = LogisticRegression()

    def train(self):
        # Train the model
        self.model.fit(self.X_train_features, self.Y_train)

    def evaluate(self):
        # Evaluate model on both training and test sets
        train_accuracy = accuracy_score(self.Y_train, self.model.predict(self.X_train_features))
        test_accuracy = accuracy_score(self.Y_test, self.model.predict(self.X_test_features))
        return {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}

    def predict(self, input_mail):
        # Predict if the input mail is spam or ham
        input_data_features = self.feature_extraction.transform(input_mail)
        prediction = self.model.predict(input_data_features)
        print(prediction[0])
        if prediction[0] == 1:
            return 'Ham mail'
            
        else :
            return 'Spam mail'

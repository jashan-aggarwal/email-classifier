import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


raw_mail_data = pd.read_csv('mail_data.csv')

# replace the null values with an empty string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Seperating the data into text and labels
X = mail_data['Message']
Y = mail_data['Category']

# test_size = 0.2 means that 20% of the data is used for testing and random_state = anyFixedNumber is used to split data in a certain fixed way every time the code runs
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)

# Transforming the text data to feature vectors that can be used as input to the Logistic Regression Model
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers so that they are not treated as strings
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
# Training using the data
model.fit(X_train_features, Y_train)

# Prediction on Training Data
# prediction_on_training_data = model.predict(X_train_features)
# accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
# print("Accuracy on Training data: ", accuracy_on_training_data)
#
# prediction_on_test_data = model.predict(X_test_features)
# accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
# print("Accuracy on Test data: ", accuracy_on_test_data)

# Saving the Trained Model
joblib.dump(model, 'spam_detection.pkl')
joblib.dump(feature_extraction, 'vectorizer.pkl')

# Loading the model for future use
model = joblib.load('spam_detection.pkl')
feature_extraction = joblib.load('vectorizer.pkl')

input_mail = input("Enter an e-mail: ")
input_mail = [input_mail]

# Converting to feature vector
input_data_features = feature_extraction.transform(input_mail)

# Making Prediction
prediction = model.predict(input_data_features)

# print(prediction)
print()

if (prediction[0] == 1):
  print("Ham Mail")
else:
  print("Spam Mail")
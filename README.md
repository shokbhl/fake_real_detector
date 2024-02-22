#A Fake News Detector
## Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import itertools
import numpy as np

Here, the code imports required libraries such as pandas for data handling, scikit-learn for machine learning operations, and other utility libraries like itertools and numpy.

# Importing dataset using pandas dataframe
df = pd.read_csv("/content/Fraud_and_real_dataset_(new) - Sheet1.csv")  # Adjust the file path

# Set index
df = df.set_index("Id")

# Handle NaN values in the 'Story' column by filling them with a placeholder string
df['Story'].fillna('No information', inplace=True)

The code reads a CSV file containing a dataset into a pandas DataFrame. It sets the 'Id' column as the index and fills any NaN values in the 'Story' column with the string 'No information'.

# Separate the labels and set up training and test datasets
y = df['Fraud?']
df.drop("Fraud?", axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df['Story'], y, test_size=0.33, random_state=53)

It separates the target variable 'Fraud?' from the features, sets up training and test datasets using the train_test_split function.

# Naive Bayes classifier for Multinomial model using Count Vectors
clf_count = MultinomialNB()
clf_count.fit(count_train, y_train)
pred_count = clf_count.predict(count_test)
score_count = metrics.accuracy_score(y_test, pred_count)
jaccard_count = metrics.jaccard_score(y_test, pred_count, pos_label='REAL', average='weighted')
print("Naive Bayes accuracy with Count Vectors: %0.3f" % score_count)
print("Jaccard Similarity with Count Vectors: %0.3f" % jaccard_count)
cm_count = metrics.confusion_matrix(y_test, pred_count, labels=unique_labels)
plot_confusion_matrix(cm_count, classes=unique_labels)
print(cm_count)


This section applies the Naive Bayes classifier to the Count Vectors. It trains the model, makes predictions, calculates accuracy, Jaccard Similarity, and the confusion matrix, then prints the results

# Naive Bayes classifier for Multinomial model using TF-IDF Vectors
clf_tfidf = MultinomialNB()
clf_tfidf.fit(tfidf_train, y_train)
pred_tfidf = clf_tfidf.predict(tfidf_test)
score_tfidf = metrics.accuracy_score(y_test, pred_tfidf)
jaccard_tfidf = metrics.jaccard_score(y_test, pred_tfidf, pos_label='REAL', average='weighted')
print("Naive Bayes accuracy with TF-IDF Vectors: %0.3f" % score_tfidf)
print("Jaccard Similarity with TF-IDF Vectors: %0.3f" % jaccard_tfidf)
cm_tfidf = metrics.confusion_matrix(y_test, pred_tfidf, labels=unique_labels)
plot_confusion_matrix(cm_tfidf, classes=unique_labels)
print(cm_tfidf)

Similar to the previous section, this applies the Naive Bayes classifier, but this time using TF-IDF Vectors.

# Applying Passive Aggressive Classifier with TF-IDF Vectors
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred_linear = linear_clf.predict(tfidf_test)
score_linear = metrics.accuracy_score(y_test, pred_linear)
jaccard_linear = metrics.jaccard_score(y_test, pred_linear, pos_label='REAL', average='weighted')
print("Passive Aggressive accuracy with TF-IDF Vectors: %0.3f" % score_linear)
print("Jaccard Similarity with TF-IDF Vectors: %0.3f" % jaccard_linear)
cm_linear = metrics.confusion_matrix(y_test, pred_linear, labels=unique_labels)
plot_confusion_matrix(cm_linear, classes=unique_labels)
print(cm_linear)
This section applies the Passive Aggressive Classifier to the TF-IDF Vectors. It trains the model, makes predictions, calculates accuracy, Jaccard Similarity, and the confusion matrix, then prints the results.

Please note that the plot_confusion_matrix function and the actual vectorization steps (e.g., count_train, tfidf_train, count_test, tfidf_test) are not provided in the code you shared, so make sure those parts are implemented accordingly.


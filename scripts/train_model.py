import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


pd.set_option('display.max_rows', None)

original_reviews = pd.read_csv('winemag-data-130k-v2.csv')
reviews = original_reviews.copy()
print(reviews.head())
# Looking at intial rows of data

print(reviews.columns)

print(reviews.dtypes)
print(reviews['points'].describe())
# Review point system is from 80 - 100, will need to consider this when classifying sentiment

print(reviews.duplicated().sum())
print(reviews.isnull().sum())
# Since our model will be using the review description and points to class the sentiment for training, missing values in other features do not impact us as the columns will be dropped

reviews = reviews[['description', 'points']]
print(reviews.isnull().sum())
print(reviews.head())
# No missing values in the features we will be using

reviews['quality'] = pd.qcut(reviews['points'], q=2, labels=['Low', 'High'])
# Classifying wine into quality classes based on point rating

plt.figure()
sns.histplot(reviews['quality'])
plt.show()
# Comparing the spread of points 

lowest_review = reviews[reviews['points'] == reviews['points'].min()]
reviews['description'] = reviews['description'].str.strip()
reviews['description'] = reviews['description'].str.lower()
reviews['description'] = reviews['description'].str.replace(r'[^a-z\s]', ' ', regex=True)
reviews['description'] = reviews['description'].str.replace(r'\s+', ' ', regex=True)
reviews = reviews.reset_index(drop=True)
# Cleaning description strings

print(lowest_review.head())
print(reviews.head())

X = reviews['description']
Y = reviews['quality']
# Separating features for model training

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, 'vectorizer.pkl')
# Training tf-idf vetorizer and using joblib so I can use the same vetorizer in another python file

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)
predictions = model.predict(X_test_tfidf)
joblib.dump(model, 'model.pkl')
# Training logistic regression model and using joblib so I can use the same vetorizer in another python file


print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
# Analysing model output and accuracy

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
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

duplicate_count = reviews.duplicated(
    subset=["description", "points"]
).sum()

print("MODEL DUPLICATE COUNT:", duplicate_count)
print("TYPE:", type(duplicate_count))

print(reviews.duplicated().sum())
print(reviews.isnull().sum())
print('check this')
reviews.duplicated(subset=["description", "points"]).sum()
# Since our model will be using the review description and points to class the sentiment for training, missing values in other features do not impact us as the columns will be dropped


duplicates = reviews.duplicated(
    subset=["description", "points"],
    keep=False
)
duplicate_rows = reviews[duplicates]
print(len(duplicate_rows))
print(duplicate_rows.head(10))
reviews = reviews.drop_duplicates(subset=["description", "points"])

print(len(reviews))
print(
    reviews.duplicated(
        subset=["description", "points"]
    ).sum()
)

reviews = reviews[['description', 'points']]
print(reviews.isnull().sum())
print(reviews.head())
# No missing values in the features we will be using

thresholds = [80, 87, 94, 101]
labels = ['Low', 'Medium', 'High']
reviews['quality'] = pd.cut(reviews['points'], bins = thresholds, labels = labels, right = False)
print(reviews['quality'].value_counts())
print(reviews['quality'].isnull().sum())
"""
# Classifying wine into quality classes based on point rating
plt.figure()
sns.histplot(reviews['quality'])
plt.show()
# Comparing the spread of points 
"""
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
Y = reviews['points']
# Separating features for model training

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, 'vectorizer.pkl')
# Training tf-idf vetorizer and using joblib so I can use the same vetorizer in another python file

model = Ridge(alpha=1.0, solver="lsqr")
model.fit(X_train_tfidf, Y_train)
predictions = model.predict(X_test_tfidf)
joblib.dump(model, 'model.pkl')
# Training logistic regression model and using joblib so I can use the same vetorizer in another python file

# Create weights from training target distribution only
score_counts = Y_train.value_counts()

sample_weights = Y_train.map(
    lambda score: 1 / score_counts[score]
)

# Normalize weights so average weight = 1
sample_weights = sample_weights / sample_weights.mean()

# Train weighted Ridge model
weighted_model = Ridge(alpha=1.0, solver="lsqr")

weighted_model.fit(
    X_train_tfidf,
    Y_train,
    sample_weight=sample_weights
)

weighted_predictions = weighted_model.predict(X_test_tfidf)


mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = root_mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
# Analysing model output and accuracy

weighted_mae = mean_absolute_error(Y_test, weighted_predictions)
weighted_mse = mean_squared_error(Y_test, weighted_predictions)
weighted_rmse = root_mean_squared_error(Y_test, weighted_predictions)
weighted_r2 = r2_score(Y_test, weighted_predictions)
print("weighted MAE:", weighted_mae)
print("weighted MSE:", weighted_mse)
print("weighted RMSE:", weighted_rmse)
print("weighted R²:", weighted_r2)


X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
analysis = pd.DataFrame()
analysis["predictions"] = predictions
analysis["description"] = X_test
analysis["actual_points"] = Y_test
analysis["residual"] = analysis["actual_points"] - analysis["predictions"]
analysis["weighted_predictions"] = weighted_predictions
analysis["weighted_residual"] = (
    analysis["actual_points"] - analysis["weighted_predictions"]
)
#analysis.to_csv("regression_prediction_analysis.csv", index=False)
print(analysis["residual"].mean())
print(analysis.groupby("actual_points")["residual"].mean())
print(analysis.groupby("actual_points")["actual_points"].count())

sample_size = analysis.groupby("actual_points")["actual_points"].count()
weighted_rmse_by_points = analysis.groupby("actual_points").apply(
    lambda group: root_mean_squared_error(
        group["actual_points"],
        group["weighted_predictions"]
    )
)

rmse_by_points = analysis.groupby("actual_points").apply(
    lambda group: root_mean_squared_error(
        group["actual_points"],
        group["predictions"]
    )
)

print(rmse_by_points)

per_group_comparison = pd.DataFrame({
    "sample_size": sample_size,
    "rmse": rmse_by_points,
    "weighted_rmse": weighted_rmse_by_points
})

print(per_group_comparison)

per_group_comparison.to_csv("per_group_comparison.csv")

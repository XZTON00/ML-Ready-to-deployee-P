import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("Social_media_impact_on_life.csv")

# Remove unwanted categories
df = df[~df['Gender'].isin(['Non-Binary', 'Prefer not to say'])]

# Missing values
print(df.isnull().mean() * 100)
df = df.dropna()

# Features and target
X = df.drop(columns=['Student_ID', 'Overall_Impact'])
y = df['Overall_Impact']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identify categorical columns
cat_cols = X_train.select_dtypes(
    include=['object', 'string', 'category']
).columns

# Categorical Pipeline
cat_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# Column Transformer
preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols)
], remainder='passthrough')

# Target Encoding
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Complete ML Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Training
model_pipeline.fit(X_train, y_train)

# Prediction
accuracy = model_pipeline.score(
    X_test,
    y_test)

print(f"Model Accuracy :", accuracy * 100, "\b%")

# y_pred = model_pipeline.predict(X_test)

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))


# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=le.classes_
# ))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Convert encoded predictions back to original labels
# y_pred_original = le.inverse_transform(y_pred)

# print("\nOriginal Predictions:")
# print(y_pred_original)

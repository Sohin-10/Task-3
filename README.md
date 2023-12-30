# Task-3
MOVIE RATING PREDICTION WITH PYTHON
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the movie dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('your_dataset.csv')

# Explore the dataset
print(df.head())

# Data preprocessing and feature engineering
# Handle missing values, drop irrelevant columns, convert categorical variables to numerical
# ...

# Visualize the data
sns.scatterplot(x='Director', y='Rating', data=df)
plt.show()

# Feature selection
X = df[['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3']]
y = df['Rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a preprocessing pipeline for categorical features
categorical_features = ['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Build the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer for Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['title_length'] = X_copy['title'].fillna("").apply(len)
        X_copy['description_length'] = X_copy['description'].fillna("").apply(len)
        X_copy['requirements_length'] = X_copy['requirements'].fillna("").apply(len)
        return X_copy[['title_length', 'description_length', 'requirements_length']]

class my_model:
    def __init__(self):
        self.numerical_columns = ["telecommuting", "has_company_logo", "has_questions"]
        self.text_columns = ["title", "location", "description", "requirements"]
        self.pipeline = None

    def fit(self, X, y):
        # Ensure numerical columns are numeric
        for col in self.numerical_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values for text columns
        X[self.text_columns] = X[self.text_columns].fillna("missing")

        # Text preprocessing using TfidfVectorizer
        text_transformers = [
            (f"{column}_tfidf", TfidfVectorizer(max_features=1000, stop_words="english"), column)
            for column in self.text_columns
        ]

        # Numerical column imputation
        numerical_transformer = ("numerical", SimpleImputer(), self.numerical_columns)

        # Feature engineering
        feature_engineering_transformer = ("feature_engineering", FeatureEngineering(), self.text_columns)

        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=text_transformers + [numerical_transformer, feature_engineering_transformer],
            sparse_threshold=0.0  # Force dense output for all transformations
        )

        # Random Forest Classifier
        model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced", n_jobs=-1)

        # Create pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ], memory="cache_dir")

        # Grid search for hyperparameter tuning
        param_grid = {
             "model__n_estimators": [100],
             "model__max_depth": [10, None],
             "model__min_samples_split": [2],
             "model__min_samples_leaf": [1],
         }
        
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=7, scoring="f1", n_jobs=-1)
        grid_search.fit(X, y)

        # Use the best model from grid search
        self.pipeline = grid_search.best_estimator_

    def predict(self, X):
        # Fill missing values for text columns
        X[self.text_columns] = X[self.text_columns].fillna("")

        # Predict using the pipeline
        return self.pipeline.predict(X)

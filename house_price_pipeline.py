import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=== 1. Data Loading ===")
    # بارگذاری دیتاست
    df = pd.read_csv('housing.csv')
    
    print("Dataset Info:")
    print(df.info())
    print(df.head())

    # جداسازی ویژگی‌ها (Features) و متغیر هدف (Target)
    y = df['median_house_value']
    X = df.drop('median_house_value', axis=1)

    print("\n=== 2. Exploratory Data Analysis & Cleaning ===")
    print("Missing values before cleaning:")
    print(X.isnull().sum())

    # پر کردن مقادیر گم‌شده در ستون total_bedrooms با میانگین (مطابق نوت‌بوک)
    X['total_bedrooms'] = X['total_bedrooms'].fillna(X['total_bedrooms'].mean())
    print("Missing values after cleaning:")
    print(X.isnull().sum())

    # تقسیم داده‌ها به بخش‌های آموزشی و آزمایشی
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n=== 3. Building Preprocessing and Modeling Pipeline ===")
    # شناسایی ستون‌های عددی و دسته‌ای
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # پایپ‌لاین پیش‌پردازش داده‌های عددی
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # پایپ‌لاین پیش‌پردازش داده‌های دسته‌ای (مانند ocean_proximity)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ترکیب پیش‌پردازنده‌ها با ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # ساخت پایپ‌لاین نهایی شامل پیش‌پردازش و مدل Random Forest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    print("\n=== 4. Hyperparameter Tuning with GridSearchCV ===")
    # تعریف گرید پارامترها برای بهینه‌سازی
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    print("\n=== 5. Model Evaluation ===")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    print("\n=== 6. Saving the Model ===")
    # ذخیره مدل نهایی با استفاده از joblib
    joblib.dump(best_model, 'house_price_model.pkl')
    print("Model successfully saved as 'house_price_model.pkl'")

if __name__ == '__main__':
    main()

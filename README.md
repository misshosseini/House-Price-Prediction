# House-Price-Prediction
An end-to-end Machine Learning pipeline to predict house prices. This project involves data preprocessing, exploratory data analysis (EDA), hyperparameter tuning using `GridSearchCV`, and building a highly accurate `RandomForestRegressor` model.

## Model Download

The pretrained model is too large for the repository. Download the model binary from the Releases: https://github.com/misshosseini/House-Price-Prediction/releases/tag/model-v1

## 🌟 Project Highlights

- **Data Preprocessing:** Handled missing values using statistical imputation (mean for numerical, mode for categorical) and transformed categorical variables using One-Hot Encoding.
- **Feature Scaling:** Standardized numerical features using `StandardScaler` to improve model convergence and performance.
- **Model Optimization:** Employed `GridSearchCV` with Cross-Validation to exhaustively search and find the optimal hyperparameters for the Random Forest model.
- **Data Visualization:** Built-in EDA and evaluation plots using `Matplotlib` and `Seaborn` to analyze price distribution, feature correlation, and model accuracy.
- **Model Deployment Ready:** Automatically exports the trained model, fitted scaler, and data column structure using `joblib` for seamless integration into production environments.

## 🛠️ Tech Stack

- **Python 3.x**
- **Scikit-Learn** (Machine Learning & Preprocessing)
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib & Seaborn** (Data Visualization)
- **Joblib** (Model Serialization)

## 📁 Project Structure

```text
├── house_data.csv               # The raw dataset (Ensure you add this to the folder)
├── house_price_pipeline.py      # The main script (Data load, train, eval, plot, and save)
├── predict.py                   # A standalone script to test new house predictions
├── README.md                    # Project documentation
# Generated after running the pipeline:
├── house_price_model.pkl    # The optimized Random Forest model
├── scaler.pkl               # The fitted StandardScaler
└── model_columns.pkl        # The final column names after encoding

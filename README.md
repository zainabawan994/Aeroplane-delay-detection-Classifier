# Flight Delay Prediction Project

## Project Overview
This project aims to predict flight delays using a dataset containing various flight-related features. The goal is to build and evaluate machine learning models (Random Forest, Decision Tree, XGBoost) to classify flights as delayed or not delayed.

## Data Source
The data is loaded from a CSV file named `airlines_delay.csv`.

## Feature Engineering
Several new features were engineered from the raw data to enhance model performance:
- **Weekend**: Binary feature indicating if the flight is on a weekend (DayOfWeek > 5).
- **Route**: Combination of `AirportFrom` and `AirportTo`.
- **BusyAirport**: Count of flights originating from `AirportFrom`.
- **AirportDelayRate**: Average delay rate from `AirportFrom`.
- **Hour**: Extracted from the `Time` column.
- **Minute**: Extracted from the `Time` column.
- **RushHour**: Binary feature indicating if the flight is during peak hours (6-9 AM or 5-9 PM).
- **NightFlight**: Binary feature indicating if the flight is during night hours (10 PM - 5 AM).
- **FlightType**: Categorical feature ('Short' if length < 120, 'Long' otherwise).
- **LongFlight**: Binary feature indicating if the flight length is 120 minutes or more.

## Exploratory Data Analysis (EDA)
EDA was performed to understand the data distribution, relationships between variables, and identify potential patterns. Key visualizations included:
- Distribution of `Class` (delayed/not delayed).
- Delay rates by `Weekend`.
- Airline-wise flight counts and delay proportions.
- Top airports and routes with highest delay rates.
- Delay rates by `Hour` and `Minute`.
- Delay rates by `FlightType` and `LongFlight`.

## Model Training
Three different machine learning models were trained and evaluated:
1.  **Random Forest Classifier**: Hyperparameter tuning was performed using GridSearchCV.
2.  **Decision Tree Classifier**: Trained with specified `max_depth` and `min_samples_split`.
3.  **XGBoost Classifier**: Trained with a custom set of hyperparameters.

### Data Preprocessing for Models
- Categorical features (`Airline`, `AirportFrom`, `AirportTo`) were one-hot encoded using `pd.get_dummies`.
- The dataset was split into training and testing sets (80% train, 20% test).

## Results

The models were evaluated based on accuracy and classification reports. The results for Decision Tree and XGBoost are presented in the notebook.

## Future Work
- Explore additional feature engineering techniques.
- Experiment with more advanced machine learning models or ensemble methods.
- Address potential class imbalance issues.
- Analyze feature importance from the trained models.

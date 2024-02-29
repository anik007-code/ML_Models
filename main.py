import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def preprocess_data(df):
    df['Start date'] = pd.to_datetime(df['Start date'])

    df['Hour_of_day'] = df['Start date'].dt.hour
    df['Day_of_week'] = df['Start date'].dt.dayofweek  # Monday is 0 and Sunday is 6

    features = df[['Start station number', 'End station number', 'Hour_of_day', 'Day_of_week']]
    target = df['Duration']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def grid_search_rf(X_train_scaled, y_train):
    rf_model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    best_params = grid_search.best_params_
    return best_params


def randomized_search_rf(X_train_scaled, y_train):
    rf_model = RandomForestRegressor(random_state=42)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='r2',
                                       random_state=42, n_jobs=-1)
    random_search.fit(X_train_scaled, y_train)
    best_params_randomized = random_search.best_params_
    return best_params_randomized


def train_and_evaluate_rf(X_train_scaled, X_test_scaled, y_train, y_test, best_params):
    best_rf_model = RandomForestRegressor(**best_params, random_state=42)
    best_rf_model.fit(X_train_scaled, y_train)
    best_rf_predictions = best_rf_model.predict(X_test_scaled)
    r2 = r2_score(y_test, best_rf_predictions)
    return best_rf_model, best_rf_predictions, r2


df = pd.read_csv('your_dataset.csv')
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

best_params_grid = grid_search_rf(X_train_scaled, y_train)

best_params_randomized = randomized_search_rf(X_train_scaled, y_train)

# Train and Evaluate with Grid Search
best_rf_model_grid, predictions_grid, r2_grid = train_and_evaluate_rf(X_train_scaled, X_test_scaled, y_train, y_test,
                                                                      best_params_grid)

# Train and Evaluate with Randomized Search
best_rf_model_randomized, predictions_randomized, r2_randomized = train_and_evaluate_rf(X_train_scaled, X_test_scaled,
                                                                                        y_train, y_test,
                                                                                        best_params_randomized)

print(f'Grid Search Best R-squared Score: {r2_grid}')
print(f'Randomized Search Best R-squared Score: {r2_randomized}')

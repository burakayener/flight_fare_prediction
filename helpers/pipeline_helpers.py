import timestamp
import pandas as pd
from eda import *
from data_prep import *
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
import warnings 
warnings.filterwarnings("ignore")

### Parameters for hyperparameter tuning

rf_params = {"max_depth": [3,5,8, None], "max_features": [3,5,7], "n_estimators": [100,200,500,1000], "min_samples_split": [2,5,10]}
gbm_params = {"learning_rate": [0.001, 0.01, 0.1, 0.05], "n_estimators": [100,200,500,1000], "max_depth": [3,5,8, None]}
et_params = {"max_depth": [3,5,8, None], "max_features": [3,5,7], "n_estimators": [100,200,500,1000], "min_samples_split": [2,5,10]}

### Regressors that will be used in the ensemble model

regressors = [('RF', RandomForestRegressor(), rf_params),
             ('GBM', GradientBoostingRegressor(), gbm_params),
             ('ET', ExtraTreesRegressor(), et_params)]


### Data Preparation Functions

def date_to_time(dataframe,col):
    dataframe[col] = pd.to_datetime(dataframe[col])


def extract_hour_min(dataframe,col):
    dataframe["NEW_" +col+ "_hour"] = dataframe[col].dt.hour
    dataframe["NEW_" + col+"_minute"] = dataframe[col].dt.minute
    dataframe.drop(col, axis=1, inplace=True)


def time_interval(time):
    if (time > 4) and (time <= 8):
        return "Early Morning"
    elif (time > 8) and (time <= 12):
        return "Morning"
    elif (time > 12) and (time <= 16):
        return "Afternoon"
    elif (time > 16) and (time <= 20):
        return "Evening"
    elif (time > 20) and (time <= 24):
        return "Night"
    elif (time >= 0) and (time <= 4):
        return "Late Night"


def duration_to_minute(dataframe,col):
    dataframe[col] = dataframe[col].str.replace("h", "*60").str.replace(" ", "+").str.replace("m", "* 1").apply(eval)


def flight_data_prep(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    dataframe.dropna(inplace=True)
    time_cols = [col for col in dataframe.columns if "Date" in col or "Time" in col]
    for time in time_cols:
        date_to_time(dataframe,time)
    dataframe["NEW_Day_of_Journey"] = dataframe["Date_of_Journey"].dt.day
    dataframe["NEW_Month_of_Journey"] = dataframe["Date_of_Journey"].dt.month
    dataframe.drop("Date_of_Journey", axis=1, inplace=True)
    for time in time_cols:
        if time != "Date_of_Journey":
            extract_hour_min(dataframe,time)
        else:
            pass
    dataframe["NEW_Dep_Time_Interval"] = list(map(lambda x: time_interval(x), dataframe["NEW_Dep_Time_hour"]))
    dataframe["NEW_Arrival_Time_Interval"] = list(map(lambda x: time_interval(x), dataframe["NEW_Arrival_Time_hour"]))
    duration_to_minute(dataframe,"Duration")
    dataframe.drop(["Additional_Info","Route"], axis=1, inplace=True)
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    one_hot_encoded_data = pd.get_dummies(dataframe, columns = ['Source'])
    OHE_source = one_hot_encoded_data.iloc[:, 13:]
    dataframe = pd.concat([dataframe, OHE_source], axis=1)
    label_airline = dataframe.groupby("Airline")["Price"].mean().sort_values(ascending=True).index
    airline_weight = {key:index for index, key in enumerate(label_airline, 0)}
    dataframe["Airline"] = dataframe["Airline"].map(airline_weight)
    dataframe["Destination"].replace("New Delhi", "Delhi", inplace=True)
    label_destination = dataframe.groupby("Destination")["Price"].mean().sort_values(ascending=True).index
    destination_weight = {key:index for index, key in enumerate(label_destination, 0)}
    dataframe["Destination"] = dataframe["Destination"].map(destination_weight)
    stops_label = {"non-stop":0, "1 stop":1, "2 stops":2, "3 stops":3, "4 stops":4}
    dataframe["Total_Stops"] = dataframe["Total_Stops"].map(stops_label)
    dep_time_index = dataframe.groupby("NEW_Dep_Time_Interval")["Price"].mean().sort_values(ascending=True).index
    dep_time_weight = {key:index for index, key in enumerate(dep_time_index, 0)}
    dataframe["NEW_Dep_Time_Interval"] = dataframe["NEW_Dep_Time_Interval"].map(dep_time_weight)
    arrival_time_index = dataframe.groupby("NEW_Arrival_Time_Interval")["Price"].mean().sort_values(ascending=True).index
    arrival_time_weight = {key:index for index, key in enumerate(arrival_time_index, 0)}
    dataframe["NEW_Arrival_Time_Interval"] = dataframe["NEW_Arrival_Time_Interval"].map(arrival_time_weight)
    low_limit, up_limit = outlier_thresholds(dataframe, "Price", q1=0.1, q3=0.9)
    dataframe["Price"] = np.where(dataframe["Price"] >up_limit, dataframe["Price"].median(),dataframe['Price'])
    dataframe.drop("Source", axis=1, inplace=True)
    X = dataframe.drop("Price", axis=1)
    y = dataframe["Price"]
    return X, y


### Base Models and Scores

def base_models(X, y, scoring="r2"):
    print("Base Models:")
    regressors = [('LR', LinearRegression()),
                  ('KNN', KNeighborsRegressor()),
                  ("SVR", SVR()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ("AdaBoost", AdaBoostRegressor()),
                  ("ET", ExtraTreesRegressor()),
                  ("AB", AdaBoostRegressor()),
                  ('GBM', GradientBoostingRegressor())]
                  
    for name, regress in regressors:
        cv_results = cross_validate(regress, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}")

### Hyperparameter Optimization


def hyperparameter_optimization(X, y, cv=3, scoring="r2"):
    print("Hyperparameter Optimization")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"############# {name}#############")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")        
              
        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)
        
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", '\n\n')
        best_models[name] = final_model
    return best_models

## Voting Regressor


def voting_regressor(X, y, best_models):
    print("Voting Regressor")

    voting_reg = VotingRegressor(
        estimators=[('RF', best_models["RF"]), ('GBM', best_models["GBM"]), ('ET', best_models["ET"])]).fit(X, y)
    scores = cross_validate(voting_reg, X, y,
                         scoring=["neg_mean_squared_error","r2","neg_mean_absolute_error", "neg_mean_absolute_percentage_error"], cv=3)
    print("Voting Regressor Scores")
    print("MSE:", -(scores["test_neg_mean_squared_error"].mean()))
    print("R2:", scores["test_r2"].mean())
    print("MAE:", -(scores["test_neg_mean_absolute_error"].mean()))
    print("MAEP:", -(scores["test_neg_mean_absolute_percentage_error"].mean()))
    return voting_reg 
# Fligh Fare Prediction 
- Dataset taken from https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh

### We have 11 diffrent variables ;
    - Airline: Airline Company
    - Date_of_Journey: Date of journey
    - Source: The place of departure
    - Destination: The place of destination 
    - Route: Route from source to destination
    - Dep_Time: Departure time
    - Arrival_Time: Arrival time
    - Duration: Duration of the flight
    - Total_Stops: Total number of stops
    - Additional_Info: Additional info for example meal not included etc. 
    - Price: Price of the ticket, also our target variable.,
 
 ### The aim of this project is ;
    - To compare diffrent regressors models. LR, KNN, SVR, CART, RF, AdaBoost, ET and GBM from 
      default parameters we compared this models. And concluded that GBM, ET, and RF algorithms 
      have fitted well our data model.
    - After that we tuned this 3 model individually.
    - Lastly we constructed a pipeline.

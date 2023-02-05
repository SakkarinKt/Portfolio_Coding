# ARIMA model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

airline = pd.read_csv("C:\\Users\\sakkarkr\\Downloads\\datasets\\Airline_passengers.csv",
                      index_col = 'Month',
                      parse_dates = True)

result = seasonal_decompose(airline['Passengers'],
                                    model = 'multiplicative')

result.plot()

from pmdarima import auto_arima
import warnings
warnings.filterswarnings('ignore')

stepwise_fit = auto_arima(airline['Passengers'], start_p = 1, start_q = 1,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action = 'ignore',
                          suppress_warnings = True,
                          stepwise = True)
stepwise_fit.summary()

# Fit ARIMA model to datasets
train = airline.iloc[:len(airline) - 12]
test = airline.iloc[len(airline) -12:]

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['Passengers'],
                order = (0, 1, 1),
                seasonal_order = (2, 1, 1, 12))

result = model.fit()
result.summary()

# Predicting against the test
start = len(train)
end = len(train) + len(test) - 1

pred = result.predict(start, end,
                      typ = 'levels').rename("Predictions")

pred.plot(legend = True)
test['Passengers'].plot(legend = True)

# Evaluate the model using MSE and RMSE
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

rmse(test['Passengers'], pred)

mean_squared_error(test['Passengers'], pred)

# Train the model on the full dataset
model = SARIMAX(airline['Passengers'],
                order = (0, 1, 1),
                seasonal_order = (2, 1, 1, 12))
result = model.fit()

# Forcast for the next 3 year
forecast = result.predict(start = len(airline),
                          end = (len(airline) - 1) + 3 * 12,
                          type = 'levels').rename('Forecast')

# Plot the forecase values
airline['Passengers'].plot(figsize = (12, 5), legend = True)
forecast.plot(legend = True)








# Ex.No-6-HOLT-WINTERS-METHOD
# Developed by: Cynthia Mehul J
# Register Number: 212223240020

## AIM:
To implement the Holt Winters Method Model using Python

## ALGORITHM:
1. Import the necessary libraries
   
2. Load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration.

3. Resample it to a monthly frequency beginning of the month.
  
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality

5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data

6. Create the final model and predict future data and plot it.

## PROGRAM:
```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = sm.datasets.sunspots.load_pandas().data

data = sm.datasets.sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)

data.plot(title='Original Data')
plt.xlabel('Year')
plt.ylabel('Sunspots')
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)
scaled_data.plot(title='Scaled Sunspots Data')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data, model="additive", period=11)
decomposition.plot()
plt.show()

scaled_data = scaled_data + 1 
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=11).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax = train_data.plot(label='Train Data')
test_data.plot(ax=ax, label='Test Data')
test_predictions_add.plot(ax=ax, label='Predictions')
ax.set_title('Train and Test Predictions')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Sunspots')
ax.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'Test RMSE: {rmse:.4f}')

print(f'Scaled data mean: {scaled_data.mean():.4f}, sqrt(variance): {np.sqrt(scaled_data.var()):.4f}')

final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=11).fit()
future_steps = 10
final_predictions = final_model.forecast(steps=future_steps)
plt.figure(figsize=(14, 6))
ax = scaled_data.plot(label='Original Data')
final_predictions.plot(ax=ax, label='Forecast')
ax.set_title('Sunspots Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Sunspots')
ax.legend()
plt.show()
```

## OUTPUT:

# Original Data Plot:

<img width="758" height="561" alt="image" src="https://github.com/user-attachments/assets/42e1c7bb-c77a-43b6-85fa-67be2536ad31" />

# Scaled Data Plot:

<img width="719" height="568" alt="image" src="https://github.com/user-attachments/assets/0c545880-4037-4624-a680-cc127cd84e07" />

# Decomposed:

<img width="823" height="590" alt="image" src="https://github.com/user-attachments/assets/be78789b-8f28-498f-84c8-806a28cc817d" />

# Predictions:

<img width="744" height="558" alt="image" src="https://github.com/user-attachments/assets/df8358a5-8e9d-418d-a06d-f63fa17a8676" />

# Forecast:

<img width="1281" height="578" alt="image" src="https://github.com/user-attachments/assets/11f1e632-9470-438a-a066-4596f1155b57" />

# Performance Metrics (Mean, Standard Deviation and RMSE):

<img width="1011" height="190" alt="image" src="https://github.com/user-attachments/assets/c991a489-7e3b-4fa7-a39f-789c6b443bd3" />

## RESULT:
Thus, the program run successfully based on the Holt Winters Method model.

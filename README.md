<H1 ALIGN =CENTER> Ex.No: 08 --  MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING... </H1>

### Date: 

### AIM :

To implement Moving Average Model and Exponential smoothing Using Python.

### ALGORITHM :

#### Step 1 :

Import necessary libraries.

#### Step 2 :

Read the AirLinePassengers data from a CSV file,Display the shape and the first 20 rows of the dataset.

#### Step 3 :

Set the figure size for plots.

#### Step 4 :

Suppress warnings.

#### Step 5 :

Plot the first 50 values of the 'Value' column.

#### Step 6 :

Perform rolling average transformation with a window size of 5.

#### Step 7 :

Display the first 10 values of the rolling mean.

#### Step 8 :

Perform rolling average transformation with a window size of 10.

#### Step 9 :

Create a new figure for plotting,Plot the original data and fitted value.

#### Step 10 :

Show the plot.

#### Step 11 :

Also perform exponential smoothing and plot the graph.
    
### PROGRAM :

#### Import the packages :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```

#### Read the Airline Passengers dataset from a CSV file :

```python
data = pd.read_csv("/content/airline.csv")
```

#### Display the shape and the first 50 rows of the dataset :

```python 
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))
```

#### Plot the first 50 values of the 'International' column :

```python
plt.plot(data['International '].head(50))
plt.title('First 50 values of the "International" column')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.show()
```

#### Perform rolling average transformation with a window size of 5 :

```python
rolling_mean_5 = data['International '].rolling(window=5).mean()
```

#### Display the first 10 values of the rolling mean :

```python
print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))
```

#### Perform rolling average transformation with a window size of 10 :

```python
rolling_mean_10 = data['International '].rolling(window=10).mean()
```

#### Plot the original data and fitted value (rolling mean with window size 10) :

```python
plt.plot(data['International '], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```

#### Fit an AutoRegressive (AR) model with 13 lags :

```python
lag_order = 13
model = AutoReg(data['International '], lags=lag_order)
model_fit = model.fit()
```

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

```python
plot_acf(data['International '])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['International '])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

#### Make predictions using the AR model :

```python
predictions = model_fit.predict(start=lag_order, end=len(data)-1)
```

#### Compare the predictions with the original data :

```python
mse = mean_squared_error(data['International '][lag_order:], predictions)
print('Mean Squared Error (MSE):', mse)
```

#### Plot the original data and predictions :

```python
plt.plot(data['International '][lag_order:], label='Original Data')
plt.plot(predictions, label='Predictions')
plt.title('AR Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```

### OUTPUT :

#### Plot the original data and fitted value :

![img1](https://github.com/anto-richard/TSA_EXP8/assets/93427534/e4d1545b-5813-4e1e-8116-fde82c6f002d)

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

![img2](https://github.com/anto-richard/TSA_EXP8/assets/93427534/c1f95c27-d035-4677-82f7-f9a79784e14e)

![img3](https://github.com/anto-richard/TSA_EXP8/assets/93427534/5a9c9016-5c42-48c5-88da-cf9e3999f519)

#### Plot the original data and predictions :

![img4](https://github.com/anto-richard/TSA_EXP8/assets/93427534/9b67d3aa-4ca2-4174-b591-a17ecb1ea76a)

### RESULT :

Thus, we have successfully implemented the Moving Average Model and Exponential smoothing using python.


---
title: "Web Application Project"
date: 2020-08-07
tags: [django, css, javascript, api, stocks data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

### **Introduction**

In this project, we "enter" a Kaggle competittion to build a model to predict NYC taxi fares. The competition expired so we can't formally submit and show our scores so here we select a subset of training data and build an ensemble model to predict the fare. This is a regression problem so our performance is measured in RMSE, or root mean-squared error. We take the differences between our predictions and the actual fares and square root of the square of this value. Taking the square of the error provides more insight than the absolute value of difference because it provides more penalty to estimates with larger error and rewards estimates with smaller error. The features provided in the training data proved to be very poor predictors of taxi fare so we had to engineer our own features. We used several distance features because distance is the predominant driver of fare prices, all of which were calculated from pickup and dropoff GPS coordinates, along with identifying airports as hotspots for more expensive fares. 

https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview

###**Reading Data**


```python
import pandas as pd
import numpy as np
import pickle
import glob
import json
from xgboost import XGBRegressor
import urllib.request
```


```python
# train_filename = "train.csv"
# chunk_size = 4000000
# separator = ","

# reader = pd.read_csv(train_filename,sep=separator,chunksize=chunk_size, low_memory=False)    


# for i, chunk in enumerate(reader):
#     out_file = "data_{}.pkl".format(i+1)
#     with open(out_file, "wb") as f:
#         pickle.dump(chunk,f,pickle.HIGHEST_PROTOCOL)

```

Above, we read in the training CSV file and break it up into smaller files so we only needed to load a very small subset of the data when training and testing our model. Ultimately we commented out the code because it takes too long to upload the train.csv file to Colab, we left it here commented out to show how we generated our Pickle file.

### **Correlation Exploration**


```python
jfk_coords = [40.64, 73.78]
ewr_coords = [40.69, 74.17]
lga_coords = [40.78, 73.87]
```

One of the most decisive factors in deciding which features to include in training our models is the feature correlation matrix. This matrix computes the covariances between every feature in our training data and then normalizes it to a coefficient between +1 and -1. +1 indicates a strong linear relationship, where one feature moves the other feature moves at exactly the same slope. A value of 0 indicates no relationship and a value of -1 indicates a polar opposite relationship. We explored to see if there was any correlation between taxi fares from transit hubs and found that there is a correlation coefficient of 0.25 between airports and taxi fares. We tried this same approach with Port Authority, Penn Station, and Grand Central but none had any positive impact on performance.

###**Added Manhattan and Euclidean Features**


```python
def manhattan_distance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
  return ((dropoff_longitude - pickup_longitude).abs() + (dropoff_latitude - pickup_latitude).abs())
```


```python
def euclidean_distance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
  return np.sqrt(((pickup_longitude-dropoff_longitude)**2) + ((pickup_latitude-dropoff_latitude)**2))
```

Here, we compute the Manhattan and Euclidean distances. Distance is the single best predictor we found for computing taxi fare with Pearson correlation coefficients above 0.8. We decided to use both Manhattan and Euclidean distances as features.


```python
def fix_negative_predictions(pred):
    for i in range(len(pred)):
        if pred[i] < 0:
            pred[i] = pred.mean()
    return pred
```

If a prediction is negative we will change it to the mean fare value. In the case of taxi fares mean estimates are more reliable than median estimates. Note that this is only used for predictions on the test data set which we don't use here because the Kaggle competition already closed.

### **Trim Data**


```python
urllib.request.urlretrieve("https://mark-test-bucket-123.s3.amazonaws.com/data_1.pkl", "data_1.pkl")

data_p_files=[]
for name in glob.glob("data_1.pkl"):
   data_p_files.append(name)

train_df = pd.DataFrame([])
for i in range(len(data_p_files)):
    train_df = train_df.append(pd.read_pickle(data_p_files[i]),ignore_index=True)
train_df.dropna(inplace=True)

```

There are too many entries in the dataset to train a model based on all 55 million of them. That would also lead to overfitting so we decided to trim it down to 4 million records for simplicity. The subset of records we are using is stored in a .pkl file hosted in an AWS S3 bucket for resiliency.

### **Coordinate Manipulations**


```python

train_df['dropoff_latitude'] = abs(train_df['dropoff_latitude'])
train_df['dropoff_longitude'] = abs(train_df['dropoff_longitude'])
train_df['pickup_latitude'] = abs(train_df['pickup_latitude'])
train_df['pickup_longitude'] = abs(train_df['pickup_longitude'])
```

We take the absolute values of all coordinates for computational simplicity. In NYC longitude coordinates are always negative and we tried many different coordinate manipulations so for performance and consistency we took the negative longitude coordinates and made them positive.


```python

train_df['rounded_pickup_longitude'] = round(train_df['pickup_longitude'], 2)
train_df['rounded_pickup_latitude'] =  round(train_df['pickup_latitude'], 2)
train_df['rounded_dropoff_longitude'] = round(train_df['dropoff_longitude'], 2)
train_df['rounded_dropoff_latitude'] =  round(train_df['dropoff_latitude'], 2)


train_df.loc[(train_df['rounded_pickup_latitude'] == jfk_coords[0]) & (train_df['rounded_pickup_longitude'] == jfk_coords[1]), 'airport'] = 1
train_df.loc[(train_df['rounded_dropoff_latitude'] == jfk_coords[0]) & (train_df['rounded_dropoff_longitude'] == jfk_coords[1]), 'airport'] = 1
train_df.loc[(train_df['rounded_pickup_latitude'] == ewr_coords[0]) & (train_df['rounded_pickup_longitude'] == ewr_coords[1]), 'airport'] = 1 
train_df.loc[(train_df['rounded_dropoff_latitude'] == ewr_coords[0]) & (train_df['rounded_dropoff_longitude'] == ewr_coords[1]), 'airport'] = 1
train_df.loc[(train_df['rounded_pickup_latitude'] == lga_coords[0]) & (train_df['rounded_pickup_longitude'] == lga_coords[1]), 'airport'] = 1
train_df.loc[(train_df['rounded_dropoff_latitude'] == lga_coords[0]) & (train_df['rounded_dropoff_longitude'] == lga_coords[1]), 'airport'] = 1 
train_df['airport'].fillna(0, inplace=True)

```

We round coordinates to 2 decimal places to capture a wider area around the airports to get as many true positives as possible when determing whether or not a fare was to/from the airport. Since fares from airports, especially JFK, were such strong indicators of higher taxi fares it is key to emphasize precision over recall.


```python

train_df['latitude_distance'] = abs(train_df['dropoff_latitude'] - train_df['pickup_latitude'])
train_df['longitude_distance'] = abs(train_df['dropoff_longitude'] - train_df['pickup_latitude'])
train_df['euclidean_distance'] = euclidean_distance(train_df['pickup_longitude'], train_df['pickup_latitude'], train_df['dropoff_longitude'], train_df['dropoff_latitude'])
train_df['manhattan_distance'] = manhattan_distance(train_df['pickup_longitude'], train_df['pickup_latitude'], train_df['dropoff_longitude'], train_df['dropoff_latitude'])


```

Here, we calculate both the Manhattan and Euclidean distances along with longitude and latitude distances. As mentioned previously, Manhattan and Euclidean distances provided strong correlation but latitude distance also has a strong correlation coefficient of just under 0.6.

### **Trim Data**


```python

upper_bound = train_df['manhattan_distance'].quantile(0.99)
lower_bound = train_df['manhattan_distance'].quantile(0.01)
train_df['manhattan_distance'] = train_df['manhattan_distance'] * 0.25
train_df['euclidean_distance'] = train_df['euclidean_distance'] * 0.25
train_df['latitude_distance'] = train_df['latitude_distance'] * 0.25
train_df['longitude_distance'] = train_df['longitude_distance'] * 0.25
train_df.drop(train_df[train_df['manhattan_distance'] > upper_bound].index, inplace=True)
train_df.drop(train_df[train_df['manhattan_distance'] < lower_bound].index, inplace=True)
train_df = train_df.replace([np.inf, -np.inf], np.nan)

```

Here we trim the top and bottom 1% of entries by Manhattan distance and scale all distance metrics down significantly to avoid weighing them too heavily relative to other features. By trimming outliers we are also enabling our model to put less emphasis on extreme values and letting it generalize much better.


```python
train_df.dropna(inplace=True)
train_df.drop(['pickup_datetime', 'key', 'dropoff_latitude', 'dropoff_longitude', 'pickup_latitude', 'pickup_longitude', 'passenger_count'], axis=1, inplace=True)
label = "fare_amount"
X_train = train_df.drop(label, axis=1)
y_train = train_df[label]
y_train=y_train.astype('float64')

reg = XGBRegressor()
reg.fit(X_train, y_train)

```

    [14:10:05] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                 silent=None, subsample=1, verbosity=1)



### **Conclusion**

Here we drop records with missing entries along with other weakly correlated features that don't improve our model. We chose over a dozen Scikit  regression models but none proved to be nearly as effective as XGBoost. It gave us a best RMSE of 3.4976 on the test data while no other models scored less than 5. XGBoost is an ensemble method which has been dominating machine learning algorithm performance and is frequently the best performer in Kaggle competitions. XGBoost stands for eXtreme Gradient Boosting, uses a gradient boosted ensemble decision tree method that trains many learners with the ultimate prediction being the prediction the most learners selected. It trades off "frills" and extra features and is focused almost exclusively on speed and performance. 

Summarizing our earlier explanations on feature selection, here is a screenshot of our feature correlation matrix, one of the most decisive factors in deciding which features to include and exclude:

![alt text](https://mark-test-bucket-123.s3.amazonaws.com/Screen+Shot+2020-07-06+at+10.12.50+AM.png)

Here is a screenshot of our Kaggle score:

![alt text](https://mark-test-bucket-123.s3.amazonaws.com/Screen+Shot+2020-07-01+at+10.32.28+AM.png)

Our top score is 3.4976, we are very pleased with these results.

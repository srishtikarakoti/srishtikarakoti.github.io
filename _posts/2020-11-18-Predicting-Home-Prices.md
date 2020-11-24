---
title: "Predicting Home Prices"
date: 2020-11-18
tags: [machine learning, eda, sklearn, data visualizations, feature engineering, linear regression, gradient boosting regression, decision tree regression, svm regression]
header:
  image: 
excerpt: "Machine Learning, EDA, Sklearn, Data Visualizations, Feature Engineering, Linear Regression, Gradient Boosting Regression, Decision Tree Regression, SVM Regression"
mathjax: "true"
---
# House Price Prediction

#### Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. The dataset used in this project proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. 

#### Purpose: With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, in this project we will predict the final price of each home.

# Part I. Data pre-processing and Exploratory Data Analysis (EDA)

In this section we import libraries to perform the associated task. i.e. pandas to load data file and dada manipulation, matplotlib & seaborn to plot heat map of distribution data and numpy to handle multi-dimensional arrays.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
import time
import warnings
import sqlite3
from sqlalchemy import create_engine
import csv
import os
warnings.filterwarnings("ignore")
import datetime as dt
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import os
```

# Loading the dataset


```python
train = pd.read_csv('/Users/srishtikarakoti/Downloads/train.csv')
test = pd.read_csv('/Users/srishtikarakoti/Downloads/test.csv')
print(train.head())
print('**'* 50)
print(test.head())
```

       Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
    0   1          60       RL         65.0     8450   Pave   NaN      Reg   
    1   2          20       RL         80.0     9600   Pave   NaN      Reg   
    2   3          60       RL         68.0    11250   Pave   NaN      IR1   
    3   4          70       RL         60.0     9550   Pave   NaN      IR1   
    4   5          60       RL         84.0    14260   Pave   NaN      IR1   
    
      LandContour Utilities    ...     PoolArea PoolQC Fence MiscFeature MiscVal  \
    0         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   
    1         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   
    2         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   
    3         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   
    4         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   
    
      MoSold YrSold  SaleType  SaleCondition  SalePrice  
    0      2   2008        WD         Normal     208500  
    1      5   2007        WD         Normal     181500  
    2      9   2008        WD         Normal     223500  
    3      2   2006        WD        Abnorml     140000  
    4     12   2008        WD         Normal     250000  
    
    [5 rows x 81 columns]
    ****************************************************************************************************
         Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
    0  1461          20       RH         80.0    11622   Pave   NaN      Reg   
    1  1462          20       RL         81.0    14267   Pave   NaN      IR1   
    2  1463          60       RL         74.0    13830   Pave   NaN      IR1   
    3  1464          60       RL         78.0     9978   Pave   NaN      IR1   
    4  1465         120       RL         43.0     5005   Pave   NaN      IR1   
    
      LandContour Utilities      ...       ScreenPorch PoolArea PoolQC  Fence  \
    0         Lvl    AllPub      ...               120        0    NaN  MnPrv   
    1         Lvl    AllPub      ...                 0        0    NaN    NaN   
    2         Lvl    AllPub      ...                 0        0    NaN  MnPrv   
    3         Lvl    AllPub      ...                 0        0    NaN    NaN   
    4         HLS    AllPub      ...               144        0    NaN    NaN   
    
      MiscFeature MiscVal MoSold  YrSold  SaleType  SaleCondition  
    0         NaN       0      6    2010        WD         Normal  
    1        Gar2   12500      6    2010        WD         Normal  
    2         NaN       0      3    2010        WD         Normal  
    3         NaN       0      6    2010        WD         Normal  
    4         NaN       0      1    2010        WD         Normal  
    
    [5 rows x 80 columns]



```python
print(train.info())
print('**'* 50)
print(test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None
    ****************************************************************************************************
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
    Id               1459 non-null int64
    MSSubClass       1459 non-null int64
    MSZoning         1455 non-null object
    LotFrontage      1232 non-null float64
    LotArea          1459 non-null int64
    Street           1459 non-null object
    Alley            107 non-null object
    LotShape         1459 non-null object
    LandContour      1459 non-null object
    Utilities        1457 non-null object
    LotConfig        1459 non-null object
    LandSlope        1459 non-null object
    Neighborhood     1459 non-null object
    Condition1       1459 non-null object
    Condition2       1459 non-null object
    BldgType         1459 non-null object
    HouseStyle       1459 non-null object
    OverallQual      1459 non-null int64
    OverallCond      1459 non-null int64
    YearBuilt        1459 non-null int64
    YearRemodAdd     1459 non-null int64
    RoofStyle        1459 non-null object
    RoofMatl         1459 non-null object
    Exterior1st      1458 non-null object
    Exterior2nd      1458 non-null object
    MasVnrType       1443 non-null object
    MasVnrArea       1444 non-null float64
    ExterQual        1459 non-null object
    ExterCond        1459 non-null object
    Foundation       1459 non-null object
    BsmtQual         1415 non-null object
    BsmtCond         1414 non-null object
    BsmtExposure     1415 non-null object
    BsmtFinType1     1417 non-null object
    BsmtFinSF1       1458 non-null float64
    BsmtFinType2     1417 non-null object
    BsmtFinSF2       1458 non-null float64
    BsmtUnfSF        1458 non-null float64
    TotalBsmtSF      1458 non-null float64
    Heating          1459 non-null object
    HeatingQC        1459 non-null object
    CentralAir       1459 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1459 non-null int64
    2ndFlrSF         1459 non-null int64
    LowQualFinSF     1459 non-null int64
    GrLivArea        1459 non-null int64
    BsmtFullBath     1457 non-null float64
    BsmtHalfBath     1457 non-null float64
    FullBath         1459 non-null int64
    HalfBath         1459 non-null int64
    BedroomAbvGr     1459 non-null int64
    KitchenAbvGr     1459 non-null int64
    KitchenQual      1458 non-null object
    TotRmsAbvGrd     1459 non-null int64
    Functional       1457 non-null object
    Fireplaces       1459 non-null int64
    FireplaceQu      729 non-null object
    GarageType       1383 non-null object
    GarageYrBlt      1381 non-null float64
    GarageFinish     1381 non-null object
    GarageCars       1458 non-null float64
    GarageArea       1458 non-null float64
    GarageQual       1381 non-null object
    GarageCond       1381 non-null object
    PavedDrive       1459 non-null object
    WoodDeckSF       1459 non-null int64
    OpenPorchSF      1459 non-null int64
    EnclosedPorch    1459 non-null int64
    3SsnPorch        1459 non-null int64
    ScreenPorch      1459 non-null int64
    PoolArea         1459 non-null int64
    PoolQC           3 non-null object
    Fence            290 non-null object
    MiscFeature      51 non-null object
    MiscVal          1459 non-null int64
    MoSold           1459 non-null int64
    YrSold           1459 non-null int64
    SaleType         1458 non-null object
    SaleCondition    1459 non-null object
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB
    None



```python
n_train = train.shape[0]
n_test = test.shape[0]
y = train['SalePrice'].values
print(train['SalePrice'].value_counts())
#print(y.value_counts())
data = pd.concat((train, test)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)
```

    140000    20
    135000    17
    145000    14
    155000    14
    190000    13
    110000    13
    160000    12
    115000    12
    139000    11
    130000    11
    125000    10
    143000    10
    185000    10
    180000    10
    144000    10
    175000     9
    147000     9
    100000     9
    127000     9
    165000     8
    176000     8
    170000     8
    129000     8
    230000     8
    250000     8
    200000     8
    141000     8
    215000     8
    148000     7
    173000     7
              ..
    64500      1
    326000     1
    277500     1
    259000     1
    254900     1
    131400     1
    181134     1
    142953     1
    245350     1
    121600     1
    337500     1
    228950     1
    274000     1
    317000     1
    154500     1
    52000      1
    107400     1
    218000     1
    104000     1
    68500      1
    94000      1
    466500     1
    410000     1
    437154     1
    219210     1
    84900      1
    424870     1
    415298     1
    62383      1
    34900      1
    Name: SalePrice, Length: 663, dtype: int64



```python
print(data.head())
print(data.shape)
```

       1stFlrSF  2ndFlrSF  3SsnPorch Alley  BedroomAbvGr BldgType BsmtCond  \
    0       856       854          0   NaN             3     1Fam       TA   
    1      1262         0          0   NaN             3     1Fam       TA   
    2       920       866          0   NaN             3     1Fam       TA   
    3       961       756          0   NaN             3     1Fam       Gd   
    4      1145      1053          0   NaN             4     1Fam       TA   
    
      BsmtExposure  BsmtFinSF1  BsmtFinSF2  ...   SaleType ScreenPorch  Street  \
    0           No       706.0         0.0  ...         WD           0    Pave   
    1           Gd       978.0         0.0  ...         WD           0    Pave   
    2           Mn       486.0         0.0  ...         WD           0    Pave   
    3           No       216.0         0.0  ...         WD           0    Pave   
    4           Av       655.0         0.0  ...         WD           0    Pave   
    
       TotRmsAbvGrd TotalBsmtSF  Utilities WoodDeckSF YearBuilt YearRemodAdd  \
    0             8       856.0     AllPub          0      2003         2003   
    1             6      1262.0     AllPub        298      1976         1976   
    2             6       920.0     AllPub          0      2001         2002   
    3             7       756.0     AllPub          0      1915         1970   
    4             9      1145.0     AllPub        192      2000         2000   
    
      YrSold  
    0   2008  
    1   2007  
    2   2008  
    3   2006  
    4   2008  
    
    [5 rows x 80 columns]
    (2919, 80)


# Data Visualization

#### Heatmap for train set


```python
plt.figure(figsize=(30,10))
sns.heatmap(train.corr(),cmap='coolwarm',annot = True)
plt.show()
```


<img src="/images/Predicting Home Prices/output_11_0.png">  


#### Pair plot


```python
sns.pairplot(train, palette='rainbow')
```




    <seaborn.axisgrid.PairGrid at 0x1a23eaaeb8>




![png](output_13_1.png)


#### CDF And Pdf for yearbuilt feature


```python
counts, bin_edges = np.histogram(train['YearBuilt'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show();
```

    [0.00616438 0.00410959 0.02534247 0.08356164 0.05684932 0.08767123
     0.17876712 0.15273973 0.09520548 0.30958904]
    [1872.  1885.8 1899.6 1913.4 1927.2 1941.  1954.8 1968.6 1982.4 1996.2
     2010. ]



![png](output_15_1.png)


#### LMPLOT for yearbuilt


```python
sns.lmplot(x='YearBuilt',y='SalePrice',data=train)
```




    <seaborn.axisgrid.FacetGrid at 0x1a4bbfdcc0>




![png](output_17_1.png)



```python
counts, bin_edges = np.histogram(train['YearBuilt'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)




plt.show();
```

    [0.00616438 0.00410959 0.02534247 0.08356164 0.05684932 0.08767123
     0.17876712 0.15273973 0.09520548 0.30958904]
    [1872.  1885.8 1899.6 1913.4 1927.2 1941.  1954.8 1968.6 1982.4 1996.2
     2010. ]



![png](output_18_1.png)


#### Box plot for GarageCars feature


```python
plt.figure(figsize=(16,8))
sns.boxplot(x='GarageCars',y='SalePrice',data=train)
plt.show()
```


![png](output_20_0.png)



```python
counts, bin_edges = np.histogram(train['GarageCars'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show();
```

    [0.05547945 0.         0.25273973 0.         0.         0.56438356
     0.         0.1239726  0.         0.00342466]
    [0.  0.4 0.8 1.2 1.6 2.  2.4 2.8 3.2 3.6 4. ]



![png](output_21_1.png)



```python
plt.figure(figsize=(16,8))
sns.barplot(x='GarageArea',y = 'SalePrice',data=train, estimator=np.mean)
plt.show()
```


![png](output_22_0.png)



```python
counts, bin_edges = np.histogram(train['GarageArea'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show();
```

    [0.05547945 0.11438356 0.20068493 0.34246575 0.16438356 0.07260274
     0.0390411  0.00684932 0.00205479 0.00205479]
    [   0.   141.8  283.6  425.4  567.2  709.   850.8  992.6 1134.4 1276.2
     1418. ]



![png](output_23_1.png)



```python
plt.figure(figsize=(16,8))
sns.barplot(x='FullBath',y = 'SalePrice',data=train)
plt.show()
```


![png](output_24_0.png)



```python
counts, bin_edges = np.histogram(train['FullBath'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show();
```

    [0.00616438 0.         0.         0.44520548 0.         0.
     0.5260274  0.         0.         0.02260274]
    [0.  0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7 3. ]



![png](output_25_1.png)



```python
sns.lmplot(x='1stFlrSF',y='SalePrice',data=train)
```




    <seaborn.axisgrid.FacetGrid at 0x1a5c3c8908>




![png](output_26_1.png)



```python
sns.boxplot(x='1stFlrSF',y='SalePrice',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a5c3c56d8>




![png](output_27_1.png)


# Feature Engineering

#### First, we have to convert all columns into numeric or categorical data.


```python
data = data[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']]
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2919 entries, 0 to 2918
    Data columns (total 21 columns):
    LotArea         2919 non-null int64
    Street          2919 non-null object
    Neighborhood    2919 non-null object
    Condition1      2919 non-null object
    Condition2      2919 non-null object
    BldgType        2919 non-null object
    HouseStyle      2919 non-null object
    OverallCond     2919 non-null int64
    Heating         2919 non-null object
    CentralAir      2919 non-null object
    Electrical      2918 non-null object
    1stFlrSF        2919 non-null int64
    2ndFlrSF        2919 non-null int64
    BsmtHalfBath    2917 non-null float64
    FullBath        2919 non-null int64
    BedroomAbvGr    2919 non-null int64
    KitchenAbvGr    2919 non-null int64
    TotRmsAbvGrd    2919 non-null int64
    GarageCars      2918 non-null float64
    GarageArea      2918 non-null float64
    PoolArea        2919 non-null int64
    dtypes: float64(3), int64(9), object(9)
    memory usage: 479.0+ KB


#### Now, fill in NULL values


```python
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean())
data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mean())
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mean())
#data['Electrical']=data['Electrical'].fillna(' ')
```

#### Onehot encoding on categorical data
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.


```python
# Categorical boolean mask
categorical_feature_mask = data.dtypes==object
# filter categorical columns using mask and turn it into alist
categorical_cols = data.columns[categorical_feature_mask].tolist()
print(categorical_cols)
print("number of categorical features ",len(categorical_cols))

# i in range(len(categorical_cols)):
 # data[i]=data[i].fillna(' ')
#data['Electrical'] = data['Electrical'].fillna(' ')
```

    ['Street', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Heating', 'CentralAir', 'Electrical']
    number of categorical features  9



```python
data = pd.get_dummies(data, columns=categorical_cols)
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2919 entries, 0 to 2918
    Data columns (total 82 columns):
    LotArea                 2919 non-null int64
    OverallCond             2919 non-null int64
    1stFlrSF                2919 non-null int64
    2ndFlrSF                2919 non-null int64
    BsmtHalfBath            2919 non-null float64
    FullBath                2919 non-null int64
    BedroomAbvGr            2919 non-null int64
    KitchenAbvGr            2919 non-null int64
    TotRmsAbvGrd            2919 non-null int64
    GarageCars              2919 non-null float64
    GarageArea              2919 non-null float64
    PoolArea                2919 non-null int64
    Street_Grvl             2919 non-null uint8
    Street_Pave             2919 non-null uint8
    Neighborhood_Blmngtn    2919 non-null uint8
    Neighborhood_Blueste    2919 non-null uint8
    Neighborhood_BrDale     2919 non-null uint8
    Neighborhood_BrkSide    2919 non-null uint8
    Neighborhood_ClearCr    2919 non-null uint8
    Neighborhood_CollgCr    2919 non-null uint8
    Neighborhood_Crawfor    2919 non-null uint8
    Neighborhood_Edwards    2919 non-null uint8
    Neighborhood_Gilbert    2919 non-null uint8
    Neighborhood_IDOTRR     2919 non-null uint8
    Neighborhood_MeadowV    2919 non-null uint8
    Neighborhood_Mitchel    2919 non-null uint8
    Neighborhood_NAmes      2919 non-null uint8
    Neighborhood_NPkVill    2919 non-null uint8
    Neighborhood_NWAmes     2919 non-null uint8
    Neighborhood_NoRidge    2919 non-null uint8
    Neighborhood_NridgHt    2919 non-null uint8
    Neighborhood_OldTown    2919 non-null uint8
    Neighborhood_SWISU      2919 non-null uint8
    Neighborhood_Sawyer     2919 non-null uint8
    Neighborhood_SawyerW    2919 non-null uint8
    Neighborhood_Somerst    2919 non-null uint8
    Neighborhood_StoneBr    2919 non-null uint8
    Neighborhood_Timber     2919 non-null uint8
    Neighborhood_Veenker    2919 non-null uint8
    Condition1_Artery       2919 non-null uint8
    Condition1_Feedr        2919 non-null uint8
    Condition1_Norm         2919 non-null uint8
    Condition1_PosA         2919 non-null uint8
    Condition1_PosN         2919 non-null uint8
    Condition1_RRAe         2919 non-null uint8
    Condition1_RRAn         2919 non-null uint8
    Condition1_RRNe         2919 non-null uint8
    Condition1_RRNn         2919 non-null uint8
    Condition2_Artery       2919 non-null uint8
    Condition2_Feedr        2919 non-null uint8
    Condition2_Norm         2919 non-null uint8
    Condition2_PosA         2919 non-null uint8
    Condition2_PosN         2919 non-null uint8
    Condition2_RRAe         2919 non-null uint8
    Condition2_RRAn         2919 non-null uint8
    Condition2_RRNn         2919 non-null uint8
    BldgType_1Fam           2919 non-null uint8
    BldgType_2fmCon         2919 non-null uint8
    BldgType_Duplex         2919 non-null uint8
    BldgType_Twnhs          2919 non-null uint8
    BldgType_TwnhsE         2919 non-null uint8
    HouseStyle_1.5Fin       2919 non-null uint8
    HouseStyle_1.5Unf       2919 non-null uint8
    HouseStyle_1Story       2919 non-null uint8
    HouseStyle_2.5Fin       2919 non-null uint8
    HouseStyle_2.5Unf       2919 non-null uint8
    HouseStyle_2Story       2919 non-null uint8
    HouseStyle_SFoyer       2919 non-null uint8
    HouseStyle_SLvl         2919 non-null uint8
    Heating_Floor           2919 non-null uint8
    Heating_GasA            2919 non-null uint8
    Heating_GasW            2919 non-null uint8
    Heating_Grav            2919 non-null uint8
    Heating_OthW            2919 non-null uint8
    Heating_Wall            2919 non-null uint8
    CentralAir_N            2919 non-null uint8
    CentralAir_Y            2919 non-null uint8
    Electrical_FuseA        2919 non-null uint8
    Electrical_FuseF        2919 non-null uint8
    Electrical_FuseP        2919 non-null uint8
    Electrical_Mix          2919 non-null uint8
    Electrical_SBrkr        2919 non-null uint8
    dtypes: float64(3), int64(9), uint8(70)
    memory usage: 473.3 KB



```python
data.shape
```




    (2919, 82)




```python
train =data[:n_train]
test = data[n_train:]
print(train.info())
print(test.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 82 columns):
    LotArea                 1460 non-null int64
    OverallCond             1460 non-null int64
    1stFlrSF                1460 non-null int64
    2ndFlrSF                1460 non-null int64
    BsmtHalfBath            1460 non-null float64
    FullBath                1460 non-null int64
    BedroomAbvGr            1460 non-null int64
    KitchenAbvGr            1460 non-null int64
    TotRmsAbvGrd            1460 non-null int64
    GarageCars              1460 non-null float64
    GarageArea              1460 non-null float64
    PoolArea                1460 non-null int64
    Street_Grvl             1460 non-null uint8
    Street_Pave             1460 non-null uint8
    Neighborhood_Blmngtn    1460 non-null uint8
    Neighborhood_Blueste    1460 non-null uint8
    Neighborhood_BrDale     1460 non-null uint8
    Neighborhood_BrkSide    1460 non-null uint8
    Neighborhood_ClearCr    1460 non-null uint8
    Neighborhood_CollgCr    1460 non-null uint8
    Neighborhood_Crawfor    1460 non-null uint8
    Neighborhood_Edwards    1460 non-null uint8
    Neighborhood_Gilbert    1460 non-null uint8
    Neighborhood_IDOTRR     1460 non-null uint8
    Neighborhood_MeadowV    1460 non-null uint8
    Neighborhood_Mitchel    1460 non-null uint8
    Neighborhood_NAmes      1460 non-null uint8
    Neighborhood_NPkVill    1460 non-null uint8
    Neighborhood_NWAmes     1460 non-null uint8
    Neighborhood_NoRidge    1460 non-null uint8
    Neighborhood_NridgHt    1460 non-null uint8
    Neighborhood_OldTown    1460 non-null uint8
    Neighborhood_SWISU      1460 non-null uint8
    Neighborhood_Sawyer     1460 non-null uint8
    Neighborhood_SawyerW    1460 non-null uint8
    Neighborhood_Somerst    1460 non-null uint8
    Neighborhood_StoneBr    1460 non-null uint8
    Neighborhood_Timber     1460 non-null uint8
    Neighborhood_Veenker    1460 non-null uint8
    Condition1_Artery       1460 non-null uint8
    Condition1_Feedr        1460 non-null uint8
    Condition1_Norm         1460 non-null uint8
    Condition1_PosA         1460 non-null uint8
    Condition1_PosN         1460 non-null uint8
    Condition1_RRAe         1460 non-null uint8
    Condition1_RRAn         1460 non-null uint8
    Condition1_RRNe         1460 non-null uint8
    Condition1_RRNn         1460 non-null uint8
    Condition2_Artery       1460 non-null uint8
    Condition2_Feedr        1460 non-null uint8
    Condition2_Norm         1460 non-null uint8
    Condition2_PosA         1460 non-null uint8
    Condition2_PosN         1460 non-null uint8
    Condition2_RRAe         1460 non-null uint8
    Condition2_RRAn         1460 non-null uint8
    Condition2_RRNn         1460 non-null uint8
    BldgType_1Fam           1460 non-null uint8
    BldgType_2fmCon         1460 non-null uint8
    BldgType_Duplex         1460 non-null uint8
    BldgType_Twnhs          1460 non-null uint8
    BldgType_TwnhsE         1460 non-null uint8
    HouseStyle_1.5Fin       1460 non-null uint8
    HouseStyle_1.5Unf       1460 non-null uint8
    HouseStyle_1Story       1460 non-null uint8
    HouseStyle_2.5Fin       1460 non-null uint8
    HouseStyle_2.5Unf       1460 non-null uint8
    HouseStyle_2Story       1460 non-null uint8
    HouseStyle_SFoyer       1460 non-null uint8
    HouseStyle_SLvl         1460 non-null uint8
    Heating_Floor           1460 non-null uint8
    Heating_GasA            1460 non-null uint8
    Heating_GasW            1460 non-null uint8
    Heating_Grav            1460 non-null uint8
    Heating_OthW            1460 non-null uint8
    Heating_Wall            1460 non-null uint8
    CentralAir_N            1460 non-null uint8
    CentralAir_Y            1460 non-null uint8
    Electrical_FuseA        1460 non-null uint8
    Electrical_FuseF        1460 non-null uint8
    Electrical_FuseP        1460 non-null uint8
    Electrical_Mix          1460 non-null uint8
    Electrical_SBrkr        1460 non-null uint8
    dtypes: float64(3), int64(9), uint8(70)
    memory usage: 236.8 KB
    None
    (1459, 82)


# Linear Regression

Linear regression is a technique for estimating linear relationships between various features and a continuous target variable. Regression means estimating a continuous real-value output. For example, if you have data that contains selling prices of houses in your city, you can estimate the selling price of your house based on that data and understand the market. Regression analysis is a subfield of Supervised Learning. Some of the questions that regression can answer  if you are dealing with housing data are as follows:

How much more can I sell my house for with an additional bedroom and bathroom?

Do houses located near malls sell for more or less than others?

What is the impact of lot size on housing prices?

Source: https://www.cloudera.com/tutorials/building-a-linear-regression-model-for-predicting-house-prices.html

We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

#### Train Test Split -
Let's split the data into a training set and a testing set. We will train the model on the training set and then use the test set to evaluate the model.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.1, random_state=101)
```


```python
# we are going to scale to data

y_train= y_train.reshape(-1,1)
y_test= y_test.reshape(-1,1)
print(X_train.info())
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1314 entries, 989 to 863
    Data columns (total 82 columns):
    LotArea                 1314 non-null int64
    OverallCond             1314 non-null int64
    1stFlrSF                1314 non-null int64
    2ndFlrSF                1314 non-null int64
    BsmtHalfBath            1314 non-null float64
    FullBath                1314 non-null int64
    BedroomAbvGr            1314 non-null int64
    KitchenAbvGr            1314 non-null int64
    TotRmsAbvGrd            1314 non-null int64
    GarageCars              1314 non-null float64
    GarageArea              1314 non-null float64
    PoolArea                1314 non-null int64
    Street_Grvl             1314 non-null uint8
    Street_Pave             1314 non-null uint8
    Neighborhood_Blmngtn    1314 non-null uint8
    Neighborhood_Blueste    1314 non-null uint8
    Neighborhood_BrDale     1314 non-null uint8
    Neighborhood_BrkSide    1314 non-null uint8
    Neighborhood_ClearCr    1314 non-null uint8
    Neighborhood_CollgCr    1314 non-null uint8
    Neighborhood_Crawfor    1314 non-null uint8
    Neighborhood_Edwards    1314 non-null uint8
    Neighborhood_Gilbert    1314 non-null uint8
    Neighborhood_IDOTRR     1314 non-null uint8
    Neighborhood_MeadowV    1314 non-null uint8
    Neighborhood_Mitchel    1314 non-null uint8
    Neighborhood_NAmes      1314 non-null uint8
    Neighborhood_NPkVill    1314 non-null uint8
    Neighborhood_NWAmes     1314 non-null uint8
    Neighborhood_NoRidge    1314 non-null uint8
    Neighborhood_NridgHt    1314 non-null uint8
    Neighborhood_OldTown    1314 non-null uint8
    Neighborhood_SWISU      1314 non-null uint8
    Neighborhood_Sawyer     1314 non-null uint8
    Neighborhood_SawyerW    1314 non-null uint8
    Neighborhood_Somerst    1314 non-null uint8
    Neighborhood_StoneBr    1314 non-null uint8
    Neighborhood_Timber     1314 non-null uint8
    Neighborhood_Veenker    1314 non-null uint8
    Condition1_Artery       1314 non-null uint8
    Condition1_Feedr        1314 non-null uint8
    Condition1_Norm         1314 non-null uint8
    Condition1_PosA         1314 non-null uint8
    Condition1_PosN         1314 non-null uint8
    Condition1_RRAe         1314 non-null uint8
    Condition1_RRAn         1314 non-null uint8
    Condition1_RRNe         1314 non-null uint8
    Condition1_RRNn         1314 non-null uint8
    Condition2_Artery       1314 non-null uint8
    Condition2_Feedr        1314 non-null uint8
    Condition2_Norm         1314 non-null uint8
    Condition2_PosA         1314 non-null uint8
    Condition2_PosN         1314 non-null uint8
    Condition2_RRAe         1314 non-null uint8
    Condition2_RRAn         1314 non-null uint8
    Condition2_RRNn         1314 non-null uint8
    BldgType_1Fam           1314 non-null uint8
    BldgType_2fmCon         1314 non-null uint8
    BldgType_Duplex         1314 non-null uint8
    BldgType_Twnhs          1314 non-null uint8
    BldgType_TwnhsE         1314 non-null uint8
    HouseStyle_1.5Fin       1314 non-null uint8
    HouseStyle_1.5Unf       1314 non-null uint8
    HouseStyle_1Story       1314 non-null uint8
    HouseStyle_2.5Fin       1314 non-null uint8
    HouseStyle_2.5Unf       1314 non-null uint8
    HouseStyle_2Story       1314 non-null uint8
    HouseStyle_SFoyer       1314 non-null uint8
    HouseStyle_SLvl         1314 non-null uint8
    Heating_Floor           1314 non-null uint8
    Heating_GasA            1314 non-null uint8
    Heating_GasW            1314 non-null uint8
    Heating_Grav            1314 non-null uint8
    Heating_OthW            1314 non-null uint8
    Heating_Wall            1314 non-null uint8
    CentralAir_N            1314 non-null uint8
    CentralAir_Y            1314 non-null uint8
    Electrical_FuseA        1314 non-null uint8
    Electrical_FuseF        1314 non-null uint8
    Electrical_FuseP        1314 non-null uint8
    Electrical_Mix          1314 non-null uint8
    Electrical_SBrkr        1314 non-null uint8
    dtypes: float64(3), int64(9), uint8(70)
    memory usage: 223.3 KB
    None


### Creating and Training the Model 


```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm)
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)


### Check Performance: Confusion, Recall, Precision matrix function


```python
# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    # This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
     
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()
    
```

### Model Evaluation 

Let's evaluate the model by checking out it's coefficients and how we can interpret them.


```python
# print the intercept
print(lm.intercept_)
```

    [0.00314264]



```python
print(lm.coef_)
```

    [[ 4.45051048e-02  9.84824511e-02  4.25260088e-01  4.27865203e-01
       1.78327828e-02  4.15521451e-02 -1.37863705e-01 -1.15439219e-01
       8.56590875e-02  1.63259801e-01 -3.61010294e-02  2.10417661e-03
      -4.02343139e+12 -4.02343139e+12 -6.67011035e+10 -2.30102049e+10
      -6.27030657e+10 -1.12926823e+11 -8.22021498e+10 -1.78620777e+11
      -1.11836064e+11 -1.48320649e+11 -1.35202864e+11 -9.09824384e+10
      -6.67011035e+10 -1.05013606e+11 -2.11155920e+11 -4.59150596e+10
      -1.28919268e+11 -9.63507828e+10 -1.31658163e+11 -1.60046570e+11
      -7.57323036e+10 -1.29840207e+11 -1.15072068e+11 -1.38633690e+11
      -7.40197272e+10 -9.50403145e+10 -5.37783063e+10  2.14622117e+11
       2.79855442e+11  4.14772784e+11  9.38391475e+10  1.44005525e+11
       8.78120842e+10  1.58195640e+11  4.70272287e+10  5.75744032e+10
       7.17805293e+11  1.60261469e+12  2.57622991e+12  7.17805293e+11
       1.01474334e+12  7.17805293e+11  7.17805293e+11  1.01474334e+12
       2.88849433e+11  1.14926148e+11  1.39179507e+11  1.34394965e+11
       2.09324318e+11  1.03979734e+12  3.21601810e+11  1.69038538e+12
       2.62989828e+11  2.93806341e+11  1.55239520e+12  5.28997847e+11
       7.05743298e+11 -5.35580280e+11 -2.85326939e+12 -2.13004879e+12
      -1.41377087e+12 -7.57136408e+11 -9.26945473e+11  9.21639721e+11
       9.21639721e+11  9.95043054e-02  5.63975093e-02  2.32233332e-02
      -9.33533279e-04  1.21084182e-01]]


### Predictions from our Model 


```python
predictions = lm.predict(X_test)
predictions= predictions.reshape(-1,1)
#plot_confusion_matrix(y_test, predictions)
```


```python
plt.figure(figsize=(15,8))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
```


![png](output_55_0.png)


### Regression Evaluation Metrics
Here are three common evaluation metrics for regression problems:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors:

1𝑛∑𝑖=1𝑛|𝑦𝑖−𝑦̂ 𝑖|

Mean Squared Error (MSE) is the mean of the squared errors:

1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2

Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)

Comparing these metrics:

MAE is the easiest to understand, because it's the average error. MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world. RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units. All of these are loss functions, because we want to minimize them.


```python
from sklearn import metrics
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#print(log_loss(y_test, predictions))
```

    MAE: 337068606006.8781
    MSE: 1.93529344563622e+24
    RMSE: 1391148247181.5217


# Gradient Boosting Regression
Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference between AdaBoost and Gradient Boosting Algorithm is how the two algorithms identify the shortcomings of weak learners (eg. decision trees). While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function (y=ax+b+e , e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data. A logical understanding of loss function would depend on what we are trying to optimise. We are trying to predict the sales prices by using a regression, then the loss function would be based off the error between true and predicted house prices.

Source:https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab


```python
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
```


```python
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
```




    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='ls', max_depth=4, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=500, n_iter_no_change=None, presort='auto',
                 random_state=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False)




```python
clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
```

    MAE: 0.3120362680874503
    MSE: 0.20317770942520502
    RMSE: 0.4507523814969867



```python
plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
plt.plot(y_test,clf_pred, c= 'blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
```


![png](output_64_0.png)



![png](output_64_1.png)


# Decision Tree Regression
The decision tree is a simple machine learning model for getting started with regression tasks.

Background - A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. (see here for more details). Not suited for large dataset because of it complexity


```python
from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 100)
dtreg.fit(X_train, y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=100, splitter='best')




```python
dtr_pred = dtreg.predict(X_test)
dtr_pred= dtr_pred.reshape(-1,1)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))
print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
```

    MAE: 0.37924813566525833
    MSE: 0.2909614276598704
    RMSE: 0.5394084052551187



```python
plt.figure(figsize=(15,8))
plt.scatter(y_test,dtr_pred,c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
plt.plot(y_test,clf_pred, c= 'blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
```


![png](output_69_0.png)



![png](output_69_1.png)


# Support Vector Machine Regression


Source: https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/


```python
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)
```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
      gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
      tol=0.001, verbose=False)




```python
svr_pred = svr.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('MSE:', metrics.mean_squared_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
```

    MAE: 0.2912518552494906
    MSE: 0.1719043690230112
    RMSE: 0.41461351765591437



```python
plt.figure(figsize=(15,8))
plt.scatter(y_test,svr_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
plt.plot(y_test,clf_pred, c= 'blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
```


![png](output_74_0.png)



![png](output_74_1.png)


# Model Comparison
##### We can determine the best working model by loking MSE rates. Hence, the best working model is Support Vector Machine. 

Let's compare error rate:


```python
error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, rfr_pred)])


print(min(metrics.mean_squared_error(y_test, predictions),min(metrics.mean_squared_error(y_test, clf_pred),min(metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, dtr_pred)))))
```

    0.1719043690230112



```python
plt.figure(figsize=(16,5))
print(error_rate)
plt.plot(error_rate)
plt.scatter(error_rate,range(1,6))
seed = 7
# prepare models
models = ['SVM','LR','BGT','DT']
```

    [1.93529345e+24 2.03177709e-01 2.90961428e-01 1.71904369e-01
     2.20636502e-01]



![png](output_77_1.png)


Now we will use test data .


```python
a = pd.read_csv('/Users/srishtikarakoti/Downloads/test.csv')
```


```python
test_id = a['Id']
print(test_id.shape)
#making dataframe 
a = pd.DataFrame(test_id, columns=['Id'])
```

    (1459,)



```python
test = sc_X.fit_transform(test)
```


```python
test.shape
```




    (1459, 82)



# Prediction with SVM Model


```python
test_prediction_svr=svr.predict(test)
test_prediction_svr= test_prediction_svr.reshape(-1,1)

test_prediction_svr

test_prediction_svr =sc_y.inverse_transform(test_prediction_svr)
test_prediction_svr
```




    array([[131952.54467848],
           [159895.57114585],
           [195871.38045739],
           ...,
           [177114.8192406 ],
           [119033.53364694],
           [235220.49806631]])




```python
test_pred_svr = pd.DataFrame(test_prediction_svr, columns=['SalePrice'])
#test_pred_svr

test_pred_svr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>131952.544678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>159895.571146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>195871.380457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>193625.194509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>194475.010263</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.concat([a,test_pred_svr], axis=1)

result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>131952.544678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>159895.571146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>195871.380457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>193625.194509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>194475.010263</td>
    </tr>
  </tbody>
</table>
</div>



### We have successfully checked data quality, inputted missing data, selected suitable features, choose at least 3 ML algorithms to build prediction models, tried to improve the model, and finally used the best model to predict house prices.

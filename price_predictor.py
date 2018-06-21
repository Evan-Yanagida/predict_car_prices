### -------------- DEFINITIONS --------------

## -------Import Dependencies-------
#Gathering data
import pandas as pd
import numpy as np
#Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
#Python built-ins
from collections import OrderedDict
import operator
import sys

## -------Define Functions-------
#Construct a multivariate variable k model
def create_model(feature_labels, df):
    # SHAPE DATA
    # Randomize order of rows in data frame.
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    #partition data
    df_length = rand_df.index.size
    partition_start = int(df_length*.75)
    train_df = df.iloc[0:partition_start]
    test_df = df.iloc[partition_start:]
    
    # TRAIN MODEL
    #init
    knn = KNeighborsRegressor(n_neighbors=12)
    
    #fit
    train_feature = df[feature_labels]
    train_target = df['price']
    knn.fit(train_feature, train_target)
    
    return knn

#A function to predict a price using premade sklearn KNN model
def predict_price(knn_model, user_data, feature_labels):
    #predict
    prediction = knn_model.predict(user_data[feature_labels])
    predicted_value = prediction[0]
    predicted_price = "$" + "%.2f" % round(predicted_value,2)   #round to two decimal places and display as a dollar amount
    
    return predicted_price

#A function to create a dictionary from two lists (that represent keys and values)
def create_dict(keys, values):
    return dict(zip(keys, values + [None] * (len(keys) - len(values))))

### -------------- PREP MODEL --------------

## -------Load in Data-------
#Read in data
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 
        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)


## -------Feature Selection-------
#Select columns to remove
drop_columns = ['make', 'fuel-type', 'aspiration', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'fuel-system',
               'num-of-doors', 'num-of-cylinders', 'symboling', 'engine-size',
               'normalized-losses']
cars = cars.drop(drop_columns, axis=1)


## -------Data Cleaning-------
#Replace all ?'s with NaN's
numeric_cars = cars.replace(to_replace='?', value=np.nan)
#Drop all rows with NaN's
numeric_cars = numeric_cars.dropna(axis=0)
#Convert all to float64
numeric_cars = numeric_cars.astype(dtype='float64')
#Normalize all values so all values range from 0 to 1, except target feature
normalized_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
normalized_cars['price'] = numeric_cars['price']

## -------Train Model-------
best_model_features = ['wheel-base', 'width', 'length', 'curb-weight', 'horsepower'] #See "Show Your Work" section to determine best features
best_model = create_model(best_model_features, normalized_cars)



### -------------- CORE FUNCTIONALITY --------------

## -------Take User Input-------
#Take in and store the user's data for their car listing, but only the features necessary to predict a price
user_listing = list()
for i in range(len(cols)):
    curr_feature = cols[i]
    if curr_feature in best_model_features:
        user_input = input(curr_feature + ": ")
        try:
            user_input = float(user_input)
        except ValueError:    #ensure the user will input numbers
            print("ERROR: Invalid input for this feature: " + "'" + curr_feature + "'")
            sys.exit()
        else:
            user_listing.append([float(user_input)])  #store user-input as a float
    else:
        user_listing.append(['?'])

## -------Convert to Data-------
#Convert the user's car listing to a format the model can process
user_data = create_dict(cols, user_listing)
user_df = pd.DataFrame(data=user_data)

## -------Predict Price-------
prediction = predict_price(best_model, user_df, best_model_features)
print("Your car is worth: " + prediction + ".")
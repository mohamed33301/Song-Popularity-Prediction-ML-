from Song_Popularity_Prediction_Classification import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#####   Loading data 
data = pd.read_csv("D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\SongPopularity_classification_test.csv", encoding='latin1')


x = data.drop(['PopularityLevel'], axis=1)

y = data['PopularityLevel']

# x = data.iloc[:, :-1].values
# y = data.iloc[:,16:17].values


print('x shape = ', x.shape)
print('y shape = ', y.shape)


#X_train, y_train = preprocessing(x, y)

X_test, y_test = preprocessing(x, y)


import pickle

with open("D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\theta_LGBMClassifier.pkl","rb") as file:

    model=pickle.load(file)


#arra=np.array(range(0,18)).reshape(1,-1)
y_prediction_=model.predict(X_test)
print('\nyour prediction is = ',y_prediction_)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_prediction_)
print("Mean squared error test:", mse)
print('score test = ',model.score(X_test,y_test)*100,' % \n')


#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
#from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



def preprocessing(X_train, y_train):
    
    # preprossessing data
    print('\n-------------------------data-------------------------------\n')
      #  Date column
    date1=pd.DataFrame({'date':X_train['Album Release Date']})
    print("\n-------------------the Album Release Date-------------------------\n")
    print(date1)
    date1[["day","month","year"]]=date1['date'].str.split("/",expand=True)
    print("\n------------------the new Album Release Date----------------------\n")
    print(date1)
    date1[["day","month","year"]].dropna(subset=["day","month","year"])

    X_train['Album Release Date']=date1["year"]
    print('\n----------------Album Release Date [year]------------------\n',X_train['Album Release Date'])


    data_train=X_train.join(y_train)


    # # check null data 
    print(data_train.isnull().sum())
    print("\n shape bfore remove null = ",data_train.shape)
    # #  fill the null in Album Release Date column by using 'mean()' method
    # lat1 = data_train['Album Release Date'].mean()
    # data_train['Album Release Date'].fillna(value=lat1, inplace=True)

    data_train.dropna(inplace=True)  

    print(data_train.isnull().sum())
    print("\n shape after remove null  = ",data_train.shape)

    # calculate persintage missing data 
    mask=data_train.isnull().any(axis=1)
    rows_nan=mask.sum()
    total_row=len(data_train)
    perc=(rows_nan/total_row)*100
    print('\nthe persintage of missing value in data = ',perc  , '\n')
    # persintage = 0.0     

    # check count for each column
    for i in data_train.select_dtypes(include='object'):    #count for each
        print(data_train[i].value_counts())


    # convert object to number
    print(data_train.info())   
    from sklearn.preprocessing import LabelEncoder
    #encode=LabelEncoder()
    def Feature_Encoder(X,cols):
        
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(X[c].values))
            X[c] = lbl.transform(list(X[c].values))
        return X

    cols=data_train.select_dtypes(include='object')
    data_train=Feature_Encoder(data_train,cols)

    print('\ndata encoder : ')
    print(data_train) 
    print(data_train.info())   


    # check outliers 
    plt.figure(figsize=(25,10))
    data_train.boxplot(color='b',sym='r+')

    print('\n-------------------------------------------------------------------------------')
    print('\nall data shape = ',data_train.shape,'\n')

    # # remove outliers
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data_train['Song Length(ms)'])
    plt.title('Boxplot of ' + 'Song Length(ms)' + ' with outliers')
    plt.show()
    # Define the lower and upper bounds for outliers
    lower_bound = 74000
    upper_bound = 360000

    # Filter the DataFrame to remove outliers
    data_train = data_train[(data_train['Song Length(ms)'] >= lower_bound) & (data_train['Song Length(ms)'] <= upper_bound)]


    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data_train['Song Length(ms)'])
    plt.title('Boxplot of ' + 'Song Length(ms)' + ' without outliers')
    plt.show()

    # check outliers 
    plt.figure(figsize=(25,10))
    data_train.boxplot(color='b',sym='r+')


    #Get the correlation between the features
    corr = data_train.corr()
    #Top 50% Correlation training features with the Value
    top_feature = corr.index[abs(corr['Popularity'])>0.0]
    #Correlation plot
    plt.subplots(figsize=(15, 10))
    top_corr = data_train[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    # i drop 4 columns becase not effict in y  ,  and small corr  , and from graph relaction shipe between y
    # Spotify Link , Song Image  , Spotify URI  ,Key  
   
    X_train = data_train.drop([ 'Song Image', 'Spotify Link', 'Spotify URI', 'Key', 'Popularity'], axis=1)
    y_train = data_train['Popularity']


    X_train= X_train.iloc[:, :].values
    y_train= y_train.values.reshape(len(y_train),-1)

    # Apply Normalization technique  data in range -1,1
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    #sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_X.fit_transform(y_train)


    print('\nX.shape = ',X_train.shape)
    print('\ny.shape = ',y_train.shape)
    
    return X_train, y_train



#####   Loading data
data = pd.read_csv("D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\SongPopularity.csv")

##understanding the data 
print(data.describe()) 
print("\n shape = ",data.shape)
print(data.head())   
print(data.tail())   
print(data.info())   
# print(sns.pairplot(data))


for i in data.select_dtypes(include='number'):
    sns.regplot(x=i,y='Popularity', data=data)
    plt.show()
    
# for i in data.select_dtypes(include='object'):
#     data[i].value_counts().plot(kind='bar',color='b')
#     plt.title(i)
#     plt.show()


x=data.drop(['Popularity'],axis=1)

y=data['Popularity']


# x = data.iloc[:, :-1].values
# y = data.iloc[:,16:17].values


print('x shape = ',x.shape)
print('y shape = ',y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=101)

X_train, y_train = preprocessing(X_train, y_train)

X_test, y_test = preprocessing(X_test, y_test)


# ml models 
print('\n-----------------------------------[GradientBoostingRegressor]---------------------------------------\n')
# model  1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model22 = GradientBoostingRegressor()
model22.fit(X_train, y_train)

y_pred21 = model22.predict(X_train)
y_pred2 = model22.predict(X_test)

mse1 = mean_squared_error(y_train, y_pred21)
print("\nMean squared error train:", mse1)

mse = mean_squared_error(y_test, y_pred2)
print("Mean squared error test:", mse)

print('score train = ',model22.score(X_train,y_train)*100,' %')
print('score test = ',model22.score(X_test,y_test)*100,' %')


print('\n-----------------------------------[RandomForestRegressor]---------------------------------------\n')

# model   2
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

predictions_train = rf.predict(X_train)
predictions_test = rf.predict(X_test)

mse2 = mean_squared_error(y_train, predictions_train)
print("\nMean squared error train:", mse2)

mse22 = mean_squared_error(y_test, predictions_test)
print("Mean squared error test:", mse22)

print('score train = ',model22.score(X_train,y_train)*100,' %')
print('score test = ',model22.score(X_test,y_test)*100,' %')


print('\n---------------------------------[LassoCV ]-----------------------------------------\n')

from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

pred1_train = lasso_cv.predict(X_train)
pred1_test = lasso_cv.predict(X_test)

mse3 = mean_squared_error(y_train, pred1_train)
print("Mean squared error train:", mse3)

mse33 = mean_squared_error(y_test, pred1_test)
print("Mean squared error test:", mse33)

print('score train = ',lasso_cv.score(X_train,y_train)*100,' %')
print('score test = ',lasso_cv.score(X_test,y_test)*100,' %')

print('\n---------------------------------[linear regression]-----------------------------------------\n')

##  model  [ LinearRegression ]
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


y_train_predicted1 = regression_model.predict(X_train)
y_pred = regression_model.predict(X_test)

mse4 = mean_squared_error(y_train, y_train_predicted1)
print("Mean squared error train:", mse4)

mse44 = mean_squared_error(y_test, y_pred)
print("Mean squared error test:", mse44)

print('score train = ',regression_model.score(X_train,y_train)*100,' %')
print('score test = ',regression_model.score(X_test,y_test)*100,' %')




print('\n---------------------------------[polynomail regression]-----------------------------------------\n')

# model   [polynomail regression]

poly_features = PolynomialFeatures(degree=3, include_bias=False)   # 2 is the best deg

X_train_poly = poly_features.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))

prediction = poly_model.predict(poly_features.fit_transform(X_test))

mse5 = mean_squared_error(y_train, y_train_predicted)
print("Mean squared error train:", mse5)

mse55 = mean_squared_error(y_test, ypred)
print("Mean squared error test:", mse55)

print('score train = ',poly_model.score(X_train_poly,y_train)*100,' %')
xtestpoly=poly_features.fit_transform(X_test)
print('score test = ',poly_model.score(xtestpoly,y_test)*100,' %')




# # show all model
# from lazypredict.Supervised import LazyRegressor 

# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None) 
# models, predictions = reg.fit(X_train, X_test, y_train, y_test) 
# print(models)


# save model 
import pickle

with open('D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\theta_RandomForestRegressor.pkl',"wb") as file:
    
    pickle.dump(rf,file)



















    
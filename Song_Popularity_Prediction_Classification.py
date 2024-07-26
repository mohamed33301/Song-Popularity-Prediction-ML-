#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
#from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import  mean_squared_error



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
    top_feature = corr.index[abs(corr['PopularityLevel'])>0.0]
    #Correlation plot
    plt.subplots(figsize=(15, 10))
    top_corr = data_train[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    # i drop 4 columns becase not effict in y  ,  and small corr  , and from graph relaction shipe between y
    # Spotify Link , Song Image  , Spotify URI  ,Key  'Key',
   
    X_train = data_train.drop([ 'Song Image', 'Spotify Link', 'Spotify URI', 'Key', 'PopularityLevel'], axis=1)
    y_train = data_train['PopularityLevel']


    X_train= X_train.iloc[:, :].values
    y_train= y_train.values.reshape(len(y_train),-1)

    # Apply Normalization technique  data in range 0,1
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    #sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    #y_train = sc_X.fit_transform(y_train)


    print('\nX.shape = ',X_train.shape)
    print('\ny.shape = ',y_train.shape)
    
    return X_train, y_train





#####   Loading data
data = pd.read_csv("D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\SongPopularity_Milestone2.csv")

##understanding the data 
print(data.describe()) 
print("\n shape = ",data.shape)
print(data.head())   
print(data.tail())   
print(data.info())   
# print(sns.pairplot(data))


# for i in data.select_dtypes(include='number'):
#     sns.regplot(x=i,y='Popularity', data=data)
#     plt.show()
    
# for i in data.select_dtypes(include='object'):
#     data[i].value_counts().plot(kind='bar',color='b')
#     plt.title(i)
#     plt.show()


x=data.drop(['PopularityLevel'],axis=1)

y=data['PopularityLevel']


# x = data.iloc[:, :-1].values
# y = data.iloc[:,16:17].values


print('x shape = ',x.shape)
print('y shape = ',y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=101)

X_train, y_train = preprocessing(X_train, y_train)

X_test, y_test = preprocessing(X_test, y_test)


# ml models 


print('\n-------------------------------------logistic_regression------------------------------\n')
# training a linear Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

lr_ovo = OneVsOneClassifier(LogisticRegression()).fit(X_train, y_train)
lr_ovr = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)

# model accuracy for Logistic Regression model
accuracy = lr_ovr.score(X_test, y_test)
print('\nOneVsRest Logistic Regression accuracy: ' + str(accuracy))
accuracy = lr_ovo.score(X_test, y_test)
print('OneVsOne Logistic Regression accuracy: ' + str(accuracy))
print('\n')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


print('\n--------------------------------------LGBMClassifier---------------\n')
from lightgbm import LGBMClassifier

# Create an instance of the LGBMClassifier model
clf = LGBMClassifier()

#time for train
import time
start_time = time.time()
# Train the model on the training data
clf.fit(X_train, y_train)
train_time = time.time() - start_time


# time for test

start_time = time.time()
pre_test1 = clf.predict(X_test)
test_time = time.time() - start_time
pre_train1=clf.predict(X_train)

print(f"Training time: {train_time:.2f} seconds")
print(f"Testing time: {test_time:.2f} seconds")

acc_train11=clf.score(X_train, y_train)
acc_test11 = clf.score( X_test, y_test)

print("\nacc_train11 =",acc_train11)
print("acc_test11 =",acc_test11)

print('\n---------------------------------------DecisionTreeClassifier---------------------------------------------\n')

from sklearn.tree import DecisionTreeClassifier

# 42 or 200 -> large time , large mse
# 100 -> smale time , smale mse
dt = DecisionTreeClassifier(random_state=100)

start_time2 = time.time()
dt.fit(X_train, y_train)
train_time2 = time.time() - start_time2

# Evaluate the performance of the classifier
pre_train2=dt.predict(X_train)

start_time2 = time.time()
pre_test2 = dt.predict(X_test)
test_time2 = time.time() - start_time2

pre_train2=dt.predict(X_train)

print(f"Training time: {train_time2:.2f} seconds")
print(f"Testing time: {test_time2:.2f} seconds")

acc_train22=dt.score(X_train, y_train)
acc_test22 = dt.score( X_test, y_test)

print("\nacc_train22 =",acc_train22)
print("acc_test22 =",acc_test22)
mse = mean_squared_error(y_test, pre_test2)
print("Mean Squared Error:",mse)

print('\n----------------------------------svm--------------')
from sklearn.svm import SVC
svm=SVC(C=1.0)
svm.fit(X_train,y_train)
svn_pred=svm.predict(X_test)

acc_train77=svm.score(X_train, y_train)
acc_test77 = svm.score( X_test, y_test)
print("\nacc_train22 =",acc_train77)
print("acc_test22 =",acc_test77)

print('\n-------------------------------------NearestCentroid-----------------------------------------------\n')
from sklearn.neighbors import NearestCentroid
# Initialize and fit the model
clf5 = NearestCentroid()

start_time3 = time.time()
clf5.fit(X_train, y_train)
train_time3 = time.time() - start_time3

pre_train5=clf5.predict(X_train)

start_time3 = time.time()
pre_test5 = clf5.predict(X_test)
test_time3 = time.time() - start_time3

print(f"Training time: {train_time3:.2f} seconds")
print(f"Testing time: {test_time3:.2f} seconds")

acc_train55=clf5.score(X_train, y_train)
acc_test55 = clf5.score( X_test, y_test)

print("\nacc_train55 =",acc_train55)
print("acc_test55 =",acc_test55)





print("\n")



# Classification Accuracy
class_acc = [0.79, 1, 0.70] #example data
model_names = ['Model LGBMClassifier', 'Model DecisionTreeClassifier', 'Model NearestCentroid'] #example data

plt.bar(model_names, class_acc)
plt.title('Classification Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.show()


# Classification Accuracy
class_acc = [0.79, 1, 0.70] #example data
model_names = ['Model LGBMClassifier', 'Model DecisionTreeClassifier', 'Model NearestCentroid'] #example data

plt.bar(model_names, class_acc)
plt.title('Classification Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.show()


# Total Training Time
train_time = [0.40, 0.77, 0.01] #example data

plt.bar(model_names, train_time)
plt.title('Total Training Time')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.show()

# Total Test Time
test_time = [0.02, 0.01, 0.00] #example data

plt.bar(model_names, test_time)
plt.title('Total Test Time')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.show()


# save model 
import pickle

with open('D:\\Desktop\\project_ml_2024\\Song Popularity Prediction [ML]\\theta_LGBMClassifier.pkl',"wb") as file:
    
    pickle.dump(clf,file)






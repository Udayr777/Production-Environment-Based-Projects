# importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from warnings import simplefilter
simplefilter('ignore')


df = pd.read_csv("C:\\Users\\udayr\\Heart Disease\\Dataset\\heart_data.csv")
df.head()   

# checking the missing values
df.isnull().sum()

# droping the column unanmed
df = df.drop("Unnamed: 0", axis=1)
df.head()

sns.lmplot(x = "biking", y = 'heart.disease', data=df)

sns.lmplot(x = 'smoking', y='heart.disease', data = df)

x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

x_df.shape

y_df.shape

x_df.head()

y_df.head()

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, train_size= 0.7, test_size=0.3, random_state=0)


# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# r2 score of the dataset
lin_pred_test = lin_model.predict(X_test)
r2_score(y_test, lin_pred_test)

# mean square error of the dataset
mean_squared_error(y_test, lin_pred_test)

print("Mean sq. error between y_test and predicted =", np.mean(lin_pred_test-y_test)**2)


# writing the pickle for serializing and deserializing the object
pickle.dump(lin_model, open('model.pkl','wb'))

# loading the pickle file 
lin_model = pickle.load(open('model.pkl','rb'))

# predict the model from pickle
print(lin_model.predict([[20.1,56.3]]))


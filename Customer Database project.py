
#We start by importing the libraries. Here we have uploaded the basic libraries to begin the project
#We will upload the machine learning libraries later
#This way its easier to keep track of uploaded libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Importing the dataset.Here I have left the document source URL empty
#The dataset source will be different for each person
customers = pd.read_csv('DATASET SOURCE')

#Most of the time its hard to understand the data by looking at it in a spreadsheet viewer.
#Before I start interacting with the data, I want to conduct some exploratory data analysis
#First, I want to know what the data is, how many columns, rows and data entries are there.
customers.info()

#Next, its a good idea to get to know the types of data that is there in our dataset
#I want to understand what the values are in the columns, whether they have categorical data or not
customers.head()

#Now, we know that there is indeed numerical data (discrete form). I want to have a look at the basic statistical makeup
customers.describe()

#After looking at the basic stat makeup. First I want to know if there is a relationship between the Time on website and the yearly amount spent
sns.jointplot(data = customers, x = 'Time on Website', y = 'Yearly Amount Spent')
#I am going to do the same with Time on App and the yearly amount spent
sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')
#From the plot there is seems to be some sort of relationship but its hard to tell what kind of relationship.

#Since I am looking at finding any quantitative relationships
#I believe that the easiest way will be to explore the relationships across the entire dataset
sns.pairplot(customers)

#From the above pairplot it seems that there is linear relationship between Yearly Amount Spent and Length of Membership
#Lets see if there is indeed a relationship between the yearly amount spent and length of membership

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data= customers)

#From the plot there is a linear relationship between Length of membership and yearly amount spent
#We are going to use linear regression to understand the quantitative relationship

#We will now import the machine learning library, declare X and y matrices and split the data into train and test set
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test size = 0.3, random_state = 101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef

#After fitting the data, lets evaluate the performance of the model
#I like to this by creating a scatterplot of predicted values vs y test
plt.scatter(y_test,predictions)
plt.xlabel('Y Test(True Values)')
plt.ylabel('Predicted Values')

#From the previous graph, it is evident that there is a linear relationship between the predicted data and the test values
#Now I am going to look at the cost functions to look at the fit of my linear model

from sklearn import metrics
from sklearn.metrics import r2_score
print('MAE', metrics.mean_absolute_error(y_test,predictions))
print('MsE', metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_sqaured_error(y_test,predictions)))

#From above, I am satisfied to find that my model is indeed accurate
#However, I want to know if my choice of model is also accurate
#For this, I am going to create a normal distribution of the residuals
#If the shape matches that of a normal distribution, then I can confirm that the choice of model is accurate
sns.distplot(y_test-predictions, bins = 50)

#Now is the time to answer the question we stated when we began analysing the data
#Should we focus more on the mobile app or web development?
cdf = pd.DataFrame(lm.coef_,X.columns,columns = ['Coeff'])
print(cdf)

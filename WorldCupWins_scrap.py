'''
import requests
#website_url = requests.get("https://www.espncricinfo.com/table/series/8039/season/1975/icc-cricket-world-cup")
from bs4 import BeautifulSoup

soup = BeautifulSoup(requests.get("https://www.espncricinfo.com/table/series/8039/season/1975/icc-cricket-world-cup").text, 'lxml')
#response = requests.get(quote_page, headers=website_url).text
#soup = BeautifulSoup(response, 'html.parser')
print(soup.prettify())
#soup = BeautifulSoup(website_url,'xml')
#print(soup.prettify())
My_table = soup.find(‘table’,{‘class’:’standings has-team-logos’})
print(My_table)
'''


'''

import requests
import lxml.html as lh
import pandas as pd

url_1='https://www.espncricinfo.com/table/series/8039/season/1975/icc-cricket-world-cup'
url_2='https://www.espncricinfo.com/table/series/8039/season/1979/icc-cricket-world-cup'
url_3='https://www.espncricinfo.com/table/series/8039/season/1983/icc-cricket-world-cup'
#Create a handle, page, to handle the contents of the website
urls=[url_1, url_2, url_3]

def data(url):
    page = requests.get(url)
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)
    #Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')


    tr_elements = doc.xpath('//tr')
    #Create empty list
    col=[]
    i=0
    #For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        #print('%d:"%s"'%(i,name))
        col.append((name,[]))


    #Since out first row is the header, data is stored on the second row onwards
    for j in range(1,len(tr_elements)):
        #T is our j'th row
        T=tr_elements[j]
        
        #If row is not of size 10, the //tr data is not from our table 
        if len(T)!=7:
            break
        
        #i is the index of our column
        i=0
        
        #Iterate through each element of the row
        for t in T.iterchildren():
            data=t.text_content() 
            #Check if row is empty
            if i>0:
            #Convert any numerical value to integers
                try:
                    data=int(data)
                except:
                    pass
            #Append the data to the empty list of the i'th column
            col[i][1].append(data)
            #Increment i for the next column
            i+=1
    #print([len(C) for (title,C) in col])
    Dict={title:column for (title,column) in col}
    df=pd.DataFrame(Dict)
    return df

    #print(df)

for i,url in enumerate(urls):   
    df=data(url)
    #print(df)
    df.to_csv('WorldCupWins{}.csv'.format(i)) 
    



'''


import pandas as pd
import numpy as numpy
data_1=pd.read_csv("/home/shri/Desktop/cricket/WorldCupWins0.csv")
data_2=pd.read_csv('/home/shri/Desktop/cricket/WorldCupWins1.csv')
data_3=pd.read_csv('/home/shri/Desktop/cricket/WorldCupWins2.csv')
############################################################################################################
data_2=data_1.append(data_2)
data=data_2.append(data_3)
#print(data_1)
data= data.drop(columns=['Unnamed: 0'])
data= data.drop([4], axis=0)
#print(data.info())#
data["M"]= data["M"].astype(int) 
data["PT"]= data["PT"].astype(int) 
data["L"]= data["L"].astype(int) 
data["W"]= data["W"].astype(int) 
data["T"]= data["T"].astype(int) 
data['N/R']=data['N/R'].astype(int)
#data= data['M'].astype(str).astype(int)

print(data.info())

data['Total_points'] = data['M'] * 4
#total_points = 12
#data["column_name_new"]=1

#print(data)
data["wining_percent"]=(data["PT"] / data['Total_points'])*100

data=data.round({ "wining_percent":2}) 

#print(data)
#print(data.info())

data.to_csv('WorldCupWins_phase_1.csv')

#############################################################################


#data_2=pd.read_csv("/home/shri/Desktop/cricket/WorldCupWins1.csv")
#data["wining_percent"]=(data["PT"]/12)*100
#print(data_2)
#data_3=pd.read_csv("/home/shri/Desktop/cricket/WorldCupWins2.csv")
#data["wining_percent"]=(data["PT"]/24)*100
#print(data_3)

#data=data_1.append(data_2)
#data=data.append(data_3)

#or
#join = data_1.join(data_2, how= "left")

#data=data.sort_index(axis = 0)

#print(data.head(5)) 
#df=data.iloc[:,[1,7]]
 
#data = data.drop([4], axis=0)
print(data)
#data.sort_values(['wining_percent'], axis=0,ascending=False, inplace=True)

data.insert(0, 'New_ID', range(0, 0 + len(data)))
data.set_index('New_ID',inplace=True)
#data.sort_values(['wining_percent'], axis=0,ascending=False, inplace=True)
#print(data)



data['wining_percent_prob'] = 0
#print(data.index)
for ind in data.index:
    if data['wining_percent'][ind] >= 60.0:
        data['wining_percent_prob'][ind] = "HIGH"
    elif 69.0 >= data['wining_percent'][ind] >= 50.0:

        data['wining_percent_prob'][ind] = "MIGHT"
    else:
        data['wining_percent_prob'][ind] = "LOW"

print(data)
#df.to_csv('WorldCupWins_phase_1.csv')


import requests 
import lxml.html as lh 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

array = data.values
X = array[:,0:7]
Y = array[:,9]
print(Y)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)
X[:,0] = labelencoder.fit_transform(X[:,0])
print(Y)
print(X)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

print(Y_validation)
print(Y_train)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))













































'''

import pandas as pd
import numpy as numpy


data=pd.read_csv("/home/shri/Desktop/cricket/WorldCupWins_phase_1.csv")


#feature enginerring 
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
data["Team_number"]= label_encoder.fit_transform(data['Team']) 
data= data.drop(columns=['Unnamed: 0'])
data.to_csv('WorldCupWins_phase_2.csv')
data=pd.read_csv("/home/shri/Desktop/cricket/WorldCupWins_phase_2.csv")
#print(data['Team_number'].unique())
print(data.columns)
#x=data["Team_number"]
#y=data["wining_percent"]





from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Split-out validation dataset
array = data.values
X = array[:,:6]
print(X)
Y = array[:,7]


# Import label encoder ---target variable is in last columns
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
X= label_encoder.fit_transform(X[:,1]) 


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
,,,





















#print(data)

#print(data.columns)

#del data["Unnamed: 0"]
#print(data.columns)



#feature enginerring 
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
data["Team_number"]= label_encoder.fit_transform(data['Team']) 
  
#print(data['Team_number'].unique())
print(data)


#simpley add one more column that will be used for winablity
#as they have played total 9 matches
#total points will be 18
#so add one more layer with the wining probabilty

data["wining_percent"]=(data["PT"]/12)*100
#print(data)

data=data.round({"Team":0, "PT":0, "Team_number":0, "wining_percent":2}) 

print(data)

#model_building
#Select input and output variables

x=data["Team_number"]
y=data["wining_percent"]

import numpy as np
from sklearn.linear_model import LinearRegression


#model = LinearRegression()
#model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

import numpy as np


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)
    '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################Data Exploration###############################

#Read files
train =pd.read_csv("train.csv")
test= pd.read_csv("test.csv")

#concat test and train set
train['source']='train'
test['source']='test'
data=pd.concat([train,test],ignore_index=True,sort=True)

#print the shape of test ,train set and data
print(train.shape,test.shape,data.shape)

#check missing values
data.apply(lambda x:sum(x.isnull()))

#structure of numerical variables
structure=data.describe(include=[np.number])

#categorical variables
data.apply(lambda x:len(x.unique()))

#Filter categorical variables
categorical_col=[col for col in data.dtypes.index if data.dtypes[col]=='object']

#Exclude id col and source
categorical_col=[col for col in categorical_col if col not in ['Item_Identifier','Outlet_Identifier','source']]

#print frequency of categories
for col in categorical_col:
    print ('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())
    
########################### Data Cleanig (Imputation)###################################
    
#Determine the avg weight per item 
item_avg_weight=data.pivot_table(values='Item_Weight',index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
missing_bool=data['Item_Weight'].isnull()

#Impute data and check missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(missing_bool))

data.loc[missing_bool,'Item_Weight'] = data.loc[missing_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])
data.loc[missing_bool,'Item_Weight'] = data.loc[missing_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])

print ('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
       
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')

#Determine avg visibility of a product
visibility_avg=data.pivot_table(values='Item_Visibility',index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool=data['Item_Visibility']==0

print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility']=data.loc[miss_bool,'Item_Identifier'].apply(lambda x:visibility_avg.loc[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

#Determine another variable with means ratio
data['Item_Visibility_MeanRatio']=data.apply(lambda x:x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']],axis=1)
print (data['Item_Visibility_MeanRatio'].describe())

#Get the first 2 char of the id
data['Item_Type_Combined']=data['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intutive categories
data['Item_Type_Combined']=data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

#Years
data['Outlet_Years']=2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#Change category of low fat
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

#Import Library
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#New variable for Outlet
data['Outlet']=le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

le=LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i].astype(str))
    
#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
data.dtypes

data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)

#Exporting Data
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Mean based
mean_sales=train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission
base1=test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales']=mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']



from sklearn import model_selection,metrics
def modelfit(alg,dtrain,dtest,predictors,target,IDcol,filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors],dtrain[target])
    
    #predict training set:
    dtrain_predictions =alg.predict(dtrain[predictors])
    
    #perform cross-validation
    cv_score=model_selection.cross_val_score(alg,dtrain[predictors],
                                              dtrain[target],cv=20,scoring='neg_mean_squared_error')

    cv_score =np.sqrt(np.abs(cv_score))
    
    #print model report
    print("\nModel Report")
    print( "RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print( "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),
                                np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Predict on testing data:
    dtest[target]=alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission=pd.DataFrame({x:dtest[x] for x in IDcol})
    submission.to_csv(filename,index=False)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
predictors=[x for x in train.columns if x not in [target]+IDcol]

#print predictors
#linear regression
alg1=LinearRegression(normalize=True)
modelfit(alg1,train,test,predictors,target,IDcol,'alg1.csv')
coef1=pd.Series(alg1.coef_,predictors).sort_values()
coef1.plot(kind='bar',title='Model Coefficients')

#Ridge regression
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')


#decision tree with just top 4 variables, a max_depth of 8 and min_samples_leaf as 150.
predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')


#Random Forest
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

#random forest with max_depth of 6 and 400 trees
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')










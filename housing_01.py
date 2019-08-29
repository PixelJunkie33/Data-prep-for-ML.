
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn


# In[6]:


import scipy


# In[7]:


print("Hello World")


# In[35]:


#import os
#import tarfile 
#from six.moves import urllib
#import os 
#os.getcwd()           ## to clal working directory 
#os.chdir(r'C:\Users\ktpra\Documents\housing_ml') #to set working directory 

Download_root = "https://github.com/ageron/handson-ml/blob/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = Download_root + "datasets/housing/housing.tgz"

def fecth_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz.path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#import pandas as pd 

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()


# In[36]:


housing.info()


# In[37]:


housing["ocean_proximity"].value_counts()


# In[38]:


housing.describe()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# In[40]:


#create a test set before going any futher using the following script 
#import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[41]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[42]:


###To ensure that the data remains consistant across multiple runs, even if you refresh the data set. The new test set will comtain 20% of the new instnaces but it will not contain and of the instances that was previously in the training set.

#housing_with_id = housing.reset_index() ###adds a index xolumn to the data frame/set #####3LIMITATION to this method is that if the data does not have a identifier column, then you will need to make sure if you use the row index as the unique identifier that, that the new data gets appended to the end of the data set, so that no data gets deleted.
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"] ##LIMITATION to this method is that some districts have the same long and lat points and will be grouped into the same train/test sets creating a sampling bias.
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id") 

from sklearn.model_selection import train_test_split ##this made the most useful method bc it still splits multiple data sets with identical number of rows it will split them on the same indices
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[43]:


print(len(train_set), "train +", len(test_set), "test")


# In[1]:


#be mindful that the median income is representive of the test set
#the following code creates a income catorgory attribute by dividing the median income by 1.5 (to limit the number of income catorgories)
#and rounding up using ceil (to have discrete catorgories) and then merging all the catorgories greater than five into catorgory five.

housing["income_cat"] = np.ceil(housing["median_income"]/ 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[47]:


housing.hist(bins=50, figsize=(20,15))


# In[63]:


# the following code performs a strafied sampling based on the income catorgory 

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
## the following code prints the results of the stratfiedshufflesplit

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[83]:


#the following code removes the income_cat attribute so the data is back to its orginal state

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[84]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[86]:


#the following code creates a copy of the data set without harming the orginal dataset

housing = strat_train_set.copy()


# In[101]:


#visuaizing geographical data ##the plot is showing high density areas
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,8),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# In[107]:



#looking for coorelations
corr_matrix = housing.corr()

#how does the attribute correlates to the housing value
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[108]:


from pandas.plotting import scatter_matrix 


# In[116]:


#check for a coorelation between attributes in pandas
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(10, 7))


# In[117]:


#narrow in on attributes that may have coorelation 

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[118]:


#experimenting and creating new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_househod"] = housing["population"]/housing["households"]


# In[119]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[121]:


#the follloing code is for preparing the data, dropping the predictors from the transormation so it isnt impacted then copy the data set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[123]:


#The folllowing code is for data the is missing values we know data does not work well with missing values 
#we have 3 options 1)get rid mof the corresponding districts
#2)get rid of the whole attribute
#3)set the values to zero, min, media etc

#######the start of training the impurter for the missing values

housing.dropna(subset=["total_bedrooms"]) #options 1 
housing.drop("total_bedrooms", axis=1) #option 2
median = housing["total_bedrooms"].median() #options3 
housing["total_bedrooms"].fillna(median, inplace=True)


#I will need to save the median value computed for the training set



# In[124]:


#####The second step too training the impurter 
the following code is the scikit method for taking care of missing values 
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

#NOTE THE MEDIAN CAN ONLY BE COMPUTED IN NUMERICAL ATTRIBUTES
#WE NEED TO CREATE A COPY OF THE DATA WITHOUT THE TEXT ATTTRIBUTE OCEAN_PROXIMITY 

housing_num = housing.drop("ocean_proximity", axis=1)

#now i can fit the imputer instances to the training data using the fit() mehtod 

imputer.fit(housing_num)


# In[126]:


#the following code below is shows the computed mean of each varialbe and stored as its statistics variable instance 
###############The results of the treined imputer
imputer.statistics_
housing_num.median().values


# In[129]:


#the following code allows me to used the trained imputer to transform the training set by replacing missing values by the leanred medians 
####KNow I transform the training impurter to transform the training set by replacing the missing values 
x = imputer.transform(housing_num)

#the result of the above lines puts the results in a plain Numpy array containing the transformed features.
##The following alows me to put the data back into a pandas dataframe. 

housing_tr = pd.DataFrame(x, columns=housing_num.columns)


# In[131]:


housing_tr.head()


# In[132]:


#the following code is usefu for handling text and catorgorical attributes 

#for example 
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)


# In[133]:


#The following code is a representation for how I can represent catorgorical attributes as numerical values 
##what is happening is a conversion of text to numbers so pandas dataframes can read, using the pandas factorize() method, 
###which maps each catorgory as different interger 

housing_cat_encoded, housing_catorgories = housing_cat.factorize()
housing_cat_encoded[:10]

#any thing with <1h ocean in the label is represented with 0
#near ocean is represted with 1
#inland is represented with 2


# In[137]:


#the following code is called one-hot codeing , it provides a one hot encoder encoder to convert interger cstorgorical valeues into one hot vectors
## Below i encode the catorgories and one hot vectors 

from sklearn.preprocessing import OneHotEncoder 
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# In[138]:


#2d arrary , to convert to dense Numpy array call the toarray()m methodd
housing_cat_1hot.toarray()


# In[156]:


#to following code shows ho to make a transofrmation in onehot, possibly more efective 
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
cat_encoder = OneHotEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[162]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[164]:


housing_cat_1hot.toarray()
cat_encoder.categories_


# In[193]:


#the following code is an example of a custome transformer pg.65-66 This has simlifies must of the process making things faster combining and creating new attributes as done in previous lines above 
#the followi ng code is the automation of a data preperation set 
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #<--This is the hyerprameter, setting it to true by default is helpful fir finding sensible parameters) #args or Kargs
               self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
               bedrooms_per_room = X[:, bedrooms_ix] / X[:rooms_ix]
               return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
               
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
               
               


# In[197]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[313]:


from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


# In[314]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[224]:


#feature scaleing and transformation pipeline 
##one of the most import transformations you can do is to apply to the data a feature scaling. The common ways are a min-max scaling and a standardization 
###Scikit-learn provides the pipeline class to help with sequences of transformations 

#####The following code below is a small pipeline for the numerical attributes 
### The following packages are for the feature scaler and standardizarion  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


housing_num_tr


# In[247]:


try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20


# In[315]:


#try:
#    from sklearn.compose import ColumnTransformer
#except ImportError:
#    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20
    
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
##feeding pandas data frame directly into the pipeline 
#the follwing code shows how, with a custom transformer 
###The dataframeselctor will transform the data by selecting the disired attributes and droppiing the rest and converting the rest
#of the dataframe to a numpy or array 

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __inti__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    


# In[317]:


##The following is a pipeline that will take a pandas Dataframe and handle numerical attributes
#try:
#    from sklearn.compose import ColumnTransformer
#except ImportError:
#    from future_encoders import ColumnTransformer
    
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([ 
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline= Pipeline([        #pipeline for catorgoical attributes
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])


# In[316]:


##But I want to join these two features into one pipeline, how? the join FeatureUnion Class
###A full pipeline handling both catorgorical and numerical attributes takes on the following strucute 

#from sklearn.pipeline import FeatureUnion

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[318]:


housing_prepared


# In[319]:


housing_prepared.shape


# In[320]:


##Training and evaluating the data set / the folllwoing code gives me a working linear regression model 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[321]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[323]:


#the following code test the training data in the regression model

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# In[324]:


# the following code measures the regression model RMSE (Root mean squared error) of the whole training set
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# the follow number below gives us a typical prediction error of 68,628.19$


# In[325]:


#the strength of this model is determiing the quality of split data. 
## Powerful model capable of finding complex nonlinear relationships in data 
###The following code will train the model with the data 

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[326]:


#the following code evaluates the training set 
###the return from the currret model shows that the model is perfect with no error
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[327]:


###The following code below shows predicts from the training set use the trained DecisionTreeRegressor 

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", tree_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# In[329]:


##the following three cells evaluate the data using a fold cross validation method 
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[330]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[331]:


#the following is an estimate of the performance of the model and also how percise the estimate is (ie std deviation)
display_scores(tree_rmse_scores)


# In[332]:


#the follwoing code codes the scores in a linear regression model

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)


# In[333]:


lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[338]:


#The following model fits the data to a RandomforestRegressor

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[341]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[343]:


#the following code performes a cross valadation of the RandomForestRegressor model

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[345]:


#the following code saves the sklearn models
from sklearn.externals import joblib

joblib.dump(housing, "housing_data.pkl")
#and later... the following code is to load the models again 
#housing_data_loaded = joblib.load("housing_data.pkl")


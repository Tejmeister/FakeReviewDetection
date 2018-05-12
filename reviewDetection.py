import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Setting up the folder paths in which the dataset is present

# In[2]:

insidendec = 'negative_polarity\\deceptive\\fold'
insidentru = 'negative_polarity\\truthful\\fold'
insidepdec = 'positive_polarity\\deceptive\\fold'
insideptru = 'positive_polarity\\truthful\\fold'

testPath   = 'TESTING'

# Initialising the lists in which the polarity, review and either it's fake or true will be stored

# In[3]:

polarity_class = []
reviews = []
spamity_class =[]

reviews_test = []
# In[4]:

pos_list = []
for data_file in sorted(os.listdir(insidendec)):
    polarity_class.append('negative')
    spamity_class.append(str(data_file.split('_')[0]))
    with open(os.path.join(insidendec, data_file)) as f:
        contents = f.read()
        reviews.append(contents)
for data_file in sorted(os.listdir(insidentru)):
    polarity_class.append('negative')
    spamity_class.append(str(data_file.split('_')[0]))
    with open(os.path.join(insidentru, data_file)) as f:
        contents = f.read()
        reviews.append(contents)
for data_file in sorted(os.listdir(insidepdec)):
    polarity_class.append('positive')
    spamity_class.append(str(data_file.split('_')[0]))
    with open(os.path.join(insidepdec, data_file)) as f:
        contents = f.read()
        reviews.append(contents)
for data_file in sorted(os.listdir(insideptru)):
    polarity_class.append('positive')
    spamity_class.append(str(data_file.split('_')[0]))
    with open(os.path.join(insideptru, data_file)) as f:
        contents = f.read()
        reviews.append(contents)


# Making the dataframe using pandas to store polarity, reviews and true or fake 

# Setting '0' for deceptive review and '1' for true review

# In[5]:

data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})

data_fm.loc[data_fm['spamity_class']=='d','spamity_class']=0
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']=1

#TESTING PART
for data_file in os.listdir(testPath):
    with open(os.path.join(testPath, data_file)) as f:
        contents = f.read()
        reviews_test.append(contents)

data_fm_test = pd.DataFrame({'review':reviews_test})

data_test_x = data_fm_test['review']

# Splitting the dataset to training and testing (0.7 and 0.3)

# In[6]:

data_x = data_fm['review']

data_y = np.asarray(data_fm['spamity_class'],dtype=int)

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)


# Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier 

# Using fit_transform() for X_train and only using transform() for X_test

# In[7]:

cv =  CountVectorizer()

X_traincv = cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)

data_test_x_cv = cv.transform(data_test_x)
# Using Naive Bayes Multinomial method as the classifier and training the data

# In[8]:

nbayes = MultinomialNB()

nbayes.fit(X_traincv, y_train)


# Predicting the fake or deceptive reviews

# using X_testcv : which is vectorized such that the dimensions are matched

# In[9]:

y_predictions = nbayes.predict(X_testcv)
y_r = nbayes.predict(data_test_x_cv)
#print(y_r)

y_result_test = list(y_r)
yres=["True" if a==1 else "Deceptive" for a in y_result_test]
for p in yres:
    print("Review is", p)
# Printing out fake or deceptive reviews

# In[10]:

y_result = list(y_predictions)
yp=["True" if a==1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review':X_testlist ,'True(1)/Deceptive(0)':yp})


# Printing out the Accuracy, Precision Score, Recall Score, F1 Score

# In[11]:

print("Accuracy: %.2f " % (metrics.accuracy_score(y_test, y_predictions)*100))
print("Precision Score: ", precision_score(y_test, y_predictions, average='micro'))
print("Recall Score: ", recall_score(y_test, y_predictions, average='micro'))
print("F1 Score: ", f1_score(y_test, y_predictions, average='micro'))
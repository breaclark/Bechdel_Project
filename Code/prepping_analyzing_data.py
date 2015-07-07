import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import cross_validation

df = pd.read_csv("../Data/complete_movies.csv")

lab = preprocessing.LabelEncoder()
df['binary'] = lab.fit_transform(df['binary'])

# making actor/writer strings back into arrays of actors/writers

def string_to_arr(thing):
    thing = string.replace(thing,"'", "")
    thing = string.replace(thing,"[","")
    thing = string.replace(thing,"]","")
    new = string.split(thing, ',')
    return new

#first set all the nans to ""

df.cast.fillna("", inplace=True)
df.writer.fillna("", inplace=True)

# because I can't store these in the dataframe, they go in their own array

actor_array = []
writer_array = [] 
for i in range (len(df)):
    actor_array.append(string_to_arr(df.ix[i,10]))
    writer_array.append(string_to_arr(df.ix[i,11]))

def make_into_dict(arr, df):
    dictionary = {}
    for i in range(len(arr)):
        #verdict is 1 for pass, 0 for fail 
        verdict = df.ix[i,4]
        #gets the cast list for movie index i 
        people = arr[i]
        for person in people:
            #puts a person in the dictionary and/or adds their passing/failing movie
            if person in dictionary:
                dictionary[person].append(verdict)
            else:
                dictionary[person] = [verdict]
    return dictionary

# Making the actual dictionaries for actors and writers

actor_pass_dict = make_into_dict(actor_array, df)
writer_pass_dict = make_into_dict(writer_array, df)


#takes the original dictionary and calculates percentage
#passing per each key in the original dictionary
#then puts this into a  new dictionary
def percentages_dictionary(orig_dict):
    percents = {}
    for person in orig_dict:
        movies = orig_dict[person]
        percents[person] = np.sum(movies)/float(len(movies))
    return percents
    
print actor_pass_dict['Mark Wahlberg']
actor_percents = percentages_dictionary(actor_pass_dict)
print actor_percents['Mark Wahlberg']


#just doing the same for the writers dictionary as well

writer_percents = percentages_dictionary(writer_pass_dict)


for i in range (len(df)):
    cast = actor_array[i]
    writers = writer_array[i]
    cast_count = 0
    cast_sum = 0
    write_count = 0
    write_sum = 0
    if cast[0] != "":
        for person in cast:
            cast_sum += actor_percents[person]*(len(actor_pass_dict[person]))
            cast_count += 1
        cast_avg = cast_sum/cast_count 
    else:
        cast_avg = 0
    if writers[0] != "":
        for writer in writers:
            write_sum += writer_percents[writer]*(len(writer_pass_dict[writer]))
            write_count += 1
        write_avg = write_sum/write_count 
    else:
        write_avg = 0
    df.ix[i,10] = cast_avg
    df.ix[i,11] = write_avg

df.to_csv('../Data/complete_movies_2.csv')

#making clean_test, binary, genre, age_rating, and director into ints

lab = preprocessing.LabelEncoder()
categoricals = ['clean_test','genre','age_rating','director']
for categorical in categoricals: df[categorical] = lab.fit_transform(df[categorical])

#making star_ratings into ints

for i in range (len(df)):
    star = df.ix[i,8]
    if not math.isnan(star):
        id = int(star)
        df.ix[i, 8] = id
    else:
        df.ix[i, 8] = -1

#puts years into year ranges

for i in range (len(df)):
    year = df.ix[i,0]
    #gets decade
    new_year = (year % 100) - (year % 10)
    #rounds new_year to the nearest half decade
    if year%10 >= 5:
        new_year += 5
    df.ix[i,0] = new_year

# Dividing budget_2013$ by 1000 so it can fit into int32 format

for i in range (len(df)):
    budget = df.ix[i,7]
    if not math.isnan(budget):
        df.ix[i,7] = budget/1000
    else:
        df.ix[i,7] = 10


print df.writer.max()
print df.writer.mean()
print df.writer.median()
print df.cast.max()
print df.cast.mean()
print df.cast.median()

for i in range (len(df)):
    if df.ix[i,10] >= 1.5: #cast avg
        df.ix[i,10] = 1
    else:
        df.ix[i,10] = 0
    if df.ix[i,11] >= .8: #writer avg
        df.ix[i,11] = 1
    else:
        df.ix[i,11] = 0


df.isnull().any()

df_new = df.drop(['imdb','title','clean_test'], axis = 1)
feature_cols = df_new.drop(['binary'], axis = 1).columns 
#all but the imdb, title, binary columns
X = df_new[feature_cols] # have to add in the other columns from arrays
y = df_new.binary

treeclf = DecisionTreeClassifier(max_depth = 2, random_state=4)
treeclf.fit(X, y)

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, 
        y, test_size=0.1, random_state=3)

treeclf.fit(X_train, y_train)
y_predictions = treeclf.predict(X_test)
print accuracy_score(y_test, y_predictions)

scores = cross_validation.cross_val_score(treeclf, X, y, cv=15)
print scores
print scores.mean()

from sklearn.metrics import (confusion_matrix, roc_curve, auc)
cm = confusion_matrix(y_test, y_predictions)
cm

target_names=['Does Not Pass', 'Passes']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, title='Confusion matrix')


proba = treeclf.predict_proba(X_test)

def plot_roc_curve(y_test, p_proba):
    # calculates: false positive rate, true positive rate, 
    fpr, tpr, thresholds = roc_curve(y_test, p_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, label= 'AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('ROC')
    plt.legend(loc="lower right")

plot_roc_curve(y_test, proba)

pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})

from sklearn.tree import export_graphviz
with open("../Data/bechdel.dot", 'wb') as f:
    f = export_graphviz(treeclf, out_file=f, feature_names=feature_cols)


from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

ran = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=3)
scores = cross_val_score(ran, X, y, cv=20, scoring='accuracy')
print format(np.mean(scores))

ada = AdaBoostClassifier(n_estimators=5, random_state=3)
scores = cross_val_score(ada, X, y, cv=20, scoring='accuracy')
print format(np.mean(scores))

from sklearn.linear_model import LogisticRegression

features_train, features_test, target_train, target_test = train_test_split(
    X, y, test_size=0.10, random_state=3)

lr = LogisticRegression()
lr.fit(features_train, target_train)

target_predicted = lr.predict(features_test)

accuracy_score(target_test, target_predicted)





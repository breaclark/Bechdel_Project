# # Bechdel Test Data Analysis

# The aim of this project will be to predict a movie's Bechdel Test verdict based on information like rating, genre, budget, cast, etc. 

# The Bechdel Test: A test consisting of 3 criteria that build on eachother, meant to test if a movie has a certain level of female representation. To pass the Bechdel Test, a movie must...
# 1. Have at least 2 named female characters
# 2. The female characters must have a conversation with each other
# 3. They must talk about something other than men
# If you want to know more about the Bechdel Test, you can find out more here https://en.wikipedia.org/wiki/Bechdel_test

# The data that I started with for this project came from https://github.com/fivethirtyeight/data/tree/master/bechdel, Fivethirtyeight's github page. The following article uses this data set to show how movies which pass the Bechdel Test tend to do better in the box office. http://fivethirtyeight.com/features/the-dollar-and-cents-case-against-hollywoods-exclusion-of-women/ I aim to build upon this data and discover new trends.

# First, here are a few charts based on the original data set.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Data/bechdel_data_min.csv")

# making a chart of number of passing movies per year

passed_per_year = []

for i in xrange(1970, 2014):
    passed = len(df[(df.binary == "PASS") & (df.year == i)])
    passed_per_year.append([i, (passed)])
    
df_ppy = pd.DataFrame(passed_per_year, columns = ['year', 'passing'])

# making a chart of percentage of passing movies per year

passed_vs_total = []

for i in xrange(1970, 2014):
    passed = len(df[(df.binary == "PASS") & (df.year == i)])
    total = len(df[df.year == i])
    passed_vs_total.append([i, (passed/float(total))*100])
    
df_pvt = pd.DataFrame(passed_vs_total, columns = ['year', 'p_passing'])

plt.subplot(2, 1, 1)
plt.bar(df_ppy.year, df_ppy.passing,1, color='deepskyblue')
plt.ylabel('Passing Movies')
plt.title('Passing Movies Per Year')

plt.subplot(2, 1, 2)
plt.bar(df_pvt.year, df_pvt.p_passing,1, color='mediumturquoise')
plt.ylabel('Percentage of Passing Movies')
plt.title('Percentage of Passing Movies Per Year')
plt.subplots_adjust(hspace=1)

# percent of total movies surveyed that totally pass, fail different parts

length = len(df)
categories = ["ok", "dubious", "nowomen", "notalk", "men"]
criterion_ps = []
for category in categories:
    criterion_ps.append((len(df[df.clean_test == category]))/float(length))


cats = ["Pass", "Dubious", "Not Enough \nFemale Characters", "Female Characters Don't \nHave a Conversation", "Female Characters \nOnly Talk About Men"]
plt.pie(criterion_ps, explode=None, labels = cats,
    colors=('springgreen', 'mediumturquoise', 'lightskyblue', 'lime', 'deepskyblue'),
    autopct='%1.1f%%', pctdistance=0.6, shadow=True,
    labeldistance=1.1, startangle=45, radius=None,
    counterclock=True, wedgeprops=None, textprops=None)
plt.axis('equal')
plt.title('Percentage of Movies Surveyed that Meet Each Bechdel Test Criterion')

# of movies that passed, what percents of each genre

passed_total = len(df[df.binary == "PASS"])
genres = ["action","comedy","drama","horror","kids","romance","scifi"]
genre_ps = []
for genre in genres:
    genre_ps.append((len(df[(df.binary == "PASS") & (df.genre == genre )]))/float(passed_total))

cats = ["Action","Comedy","Drama","Horror","Kids","Romance","Science Fiction"]
plt.pie(genre_ps, explode=None, labels = cats,
    colors=('springgreen', 'mediumturquoise', 'lightskyblue', 'lime', 'deepskyblue', 'slategray','dodgerblue'),
    autopct='%1.1f%%', pctdistance=0.6, shadow=True,
    labeldistance=1.1, startangle=-15, radius=None,
    counterclock=True, wedgeprops=None, textprops=None)
plt.axis('equal')
plt.title('Genre Breakdown of Passing Movies')

# of movies that passed, what age rating of each movie

ratings = ["G","PG","PG13","R","NR"]
rating_ps = []
for rating in ratings:
    rating_ps.append((len(df[(df.binary == "PASS") & (df.age_rating == rating )]))/float(passed_total))

plt.pie(rating_ps, explode=None, labels = ratings,
    colors=('springgreen', 'mediumturquoise', 'lime', 'lightskyblue', 'deepskyblue'),
    autopct='%1.1f%%', pctdistance=0.6, shadow=True,
    labeldistance=1.1, startangle=None, radius=None,
    counterclock=True, wedgeprops=None, textprops=None)
plt.axis('equal')
plt.title('Age Rating Breakdown of Passing Movies')



















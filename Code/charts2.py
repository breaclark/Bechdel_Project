import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/complete_movies.csv")

# making a chart for imdb star percentages for passing movies

for i in range (len(df)):
    thing = df.ix[i,8]
    if not math.isnan(thing):
        id = int(thing)
        df.ix[i, 8] = id

passing = df[df.binary == "PASS"]
categories = [0,1,2,3,4,5,6,7,8,9]
criterion_ps = []
for category in categories:
    criterion_ps.append(len(passing[passing.star_rating == category]))

cats = ["0 stars", "1 star", "2 stars", "3 stars", "4 stars", "5 stars", "6 stars", "7 stars", "8 stars", "9 stars"]
plt.bar(np.arange(10), criterion_ps, color='dodgerblue')
plt.title('IMDB Star Rating Distribution for Passing Movies')

# making a chart for imdb star_ratings for general movies surveyed

categories = [0,1,2,3,4,5,6,7,8,9]
criterion_ps = []
for category in categories:
    criterion_ps.append(len(df[df.star_rating == category]))

plt.bar(np.arange(10), criterion_ps, color='mediumturquoise')
plt.title('IMDB Star Rating Distribution for All Movies Surveyed')

df_2 = pd.read_csv("Data/complete_movies_2.csv")

# The writers with the highest writer score in the dataframe

top_writers = writer_array[df_2.writer.argmax()]
print "The movie with the highest writers score was %s" % (df_2.title[df_2.writer.argmax()])
for person in top_writers:
    print "%s has %.2f movie passes out of %s movies" % (person , writer_percents[person], len(writer_pass_dict[person]))

print "Also the movie's binary score was %s" % (df_2.binary[df_2.writer.argmax()])

#just a chart to show how writers score and binary correlate

df_3 = df_2.drop(['Unnamed: 0','year','imdb','clean_test','genre','age_rating','budget_2013$','star_rating','director', 'cast'], axis = 1)

for i in range (len(df_3)):
    df_3.ix[i, 'Writers score > 0.8?'] = (df_3.ix[i,2] > 0.8)
df_3

# making a chart for writer score distribution
plt.plot(df_3.writer, df_3.binary, 'cx')
plt.axis([0, 8, -.5, 1.5])
plt.title('Writer Average vs. Binary Score (0 = pass, 1 = fail)')

# making a chart for cast score distribution
plt.plot(df_2.cast, df_2.binary, 'bx')
plt.axis([0, 6, -.5, 1.5])
plt.title('Cast Average vs. Binary Score (0 = fail, 1 = pass)')

Avengers_writers = ['Joss Whedon', 'Stan Lee']
Avengers_cast = ['Robert Downey Jr.', 'Chris Hemsworth', 'Mark Ruffalo', 'Chris Evans', 'Scarlett Johansson']

Interstellar_writers = ['Jonathan Nolan', 'Christopher Nolan']
Interstellar_cast = ['Matthew McConaughey', 'Anne Hathaway', 'Jessica Chastain']

Apes_writers = ['Mark Bomback']
Apes_cast = ['Andy Serkis', 'Jason Clarke', 'Gary Oldman']

Edge_writers = ['Christopher McQuarrie','Jez Butterworth' ]
Edge_cast = ['Tom Cruise', 'Emily Blunt', 'Bill Paxton']

def test_writer(writers):
    write_sum = 0
    write_count = 0
    for writer in writers:
        write_sum += writer_percents[writer]*(len(writer_pass_dict[writer]))
        write_count += 1
    return write_sum/write_count 

print "Avenger writer score: %s" % (test_writer(Avengers_writers))
print "Interstellar writer score: %s" % (test_writer(Interstellar_writers))
print "Apes writer score: %s" % (test_writer(Apes_writers))
print "Edge writer score: %s" % (test_writer(Edge_writers))

def test_cast(cast):
    cast_sum = 0
    cast_count = 0
    for actor in cast:
        if actor in actor_pass_dict:
            cast_sum += actor_percents[actor]*(len(actor_pass_dict[actor]))
            cast_count += 1
    return cast_sum/cast_count 

print "Avenger cast score: %s" % (test_cast(Avengers_cast))
print "Interstellar cast score: %s" % (test_cast(Interstellar_cast))
print "Apes cast score: %s" % (test_cast(Apes_cast))
print "Edge cast score: %s" % (test_cast(Edge_cast))


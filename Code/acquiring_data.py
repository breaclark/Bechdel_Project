import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imdb import IMDb
im = IMDb()

df = pd.read_csv("../Data/bechdel_data_min.csv")

#just a little test to show imdbpy works
id = df.iloc[4][1] # snagging the movie id for the movie "42"
real_id = int(id[2:]) # converting to an int
movie = im.get_movie(real_id) #getting the movie object from imdbpy
df.loc[4, 'star_rating'] = movie['rating']# setting the cell
df.head()

movies = dict()
for i in range (len(df)):
    id = df.iloc[i][1]
    real_id = int(id[2:])
    movie = im.get_movie(real_id)
    movies[real_id] = movie
    
def add_zeros(id_number):
   str_id = str(id_number)  
   while len(str_id) < 7: 
       str_id = '0' + str_id     
   return str_id

def arr_to_string(thing):
   new = []
   for i in thing:
       new.append(str(i))
   return str(new)

for movie_id in movies:
   real_movie = movies[movie_id]   
   df_id = 'tt' + add_zeros(movie_id) 
   if len(df[df.imdb == df_id].index.values) == 0:
       continue
   i = df[df.imdb == df_id].index.values[0]
   df.ix[i, 'star_rating'] = real_movie.get('rating')   
   df.ix[i, 'director'] = real_movie['directed by'][0]  
   df.ix[i, 'cast'] = arr_to_string(real_movie['actors'][:5])
   #df.ix[i, 'writer'] = arr_to_string(real_movie.get('writer'))
   df.ix[i, 'genre'] = real_movie['genre'][0]

df.to_csv('../Data/complete_movies.csv')
import numpy as np
import pandas as pd

class data_handler():
    
    def __init__(self,path):
        self.path = path


    def get_names(self):

        movie_name = np.asarray(pd.read_csv(self.path+'movies.csv')[['title']])
#         movie_ID = np.asarray(pd.read_csv(self.path+'movies.csv')[['movieId']])
        return movie_name

    def read_data(self,plot=False):

        ratings = pd.read_csv(self.path+'ratings.csv')
        userId = np.asarray(ratings[['userId']])
        movieId = np.asarray(ratings[['movieId']])
        rating = np.asarray(ratings[['rating']])



        n_users = ratings.userId.unique() #find unique userId's
        n_items = ratings.movieId.unique() #find unique movieId's
        n_rating = ratings.rating.unique() #find number of unique ratings

        keys = n_items
        values = np.arange(len(n_items))
        dictionary = dict(zip(keys, values)) # dictionary maps index to movieId
        data = np.zeros((len(n_users), len(n_items)))
        # rows correspond to userId and column correspond to movieId by the n_users and n_items arrays


        for i in range(len(ratings)):
            movie_loc = dictionary.get(movieId[i][0])
            user_loc = userId[i]-1
            loc_rating = rating[i]
            data[user_loc,movie_loc] = loc_rating

        userId = np.arange(len(n_users))
        np.random.seed(1)
        np.random.shuffle(data)
        np.random.seed(1)
        np.random.shuffle(userId)

        if plot:
            import matplotlib.pyplot as plt;plt.rcdefaults()
            import matplotlib.pyplot as plt

            values = []
            n_rating = np.sort(n_rating)
            for i in range(len(n_rating)):
                values = np.append(values,np.count_nonzero(rating == n_rating[i]))


            plt.bar(np.arange(len(n_rating)), values, align='center', alpha=0.5)
            plt.xticks(np.arange(len(n_rating)), n_rating, rotation='vertical')
            plt.ylabel('No. of ratings')
            plt.xlabel("Rating")
            plt.title('No. of each rating')
            plt.show()

        return data,movieId

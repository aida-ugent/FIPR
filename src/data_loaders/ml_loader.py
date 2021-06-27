import numpy as np
import pandas as pd

from .data_loader import DataLoader


class Movielens100kLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "ml-100k"

    def get_sens_attr_name(self):
        return "age"

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': True
        }

    def _load(self):
        data = self._load_user_movie_data()

        # Throw away rating information.
        data = data[:, [0, 1, 3]]

        # Convert matrix to labelled entities.
        positive_edges = np.array([["t0user_" + str(row[0]), "t1movie_" + str(row[1])] for row in data])

        attributes = self.__load_attributes()
        return positive_edges, attributes

    def __load_attributes(self):
        attributes_dict = {}

        # Load user data.
        users = self._load_user_data()
        for user in users:
            user_prepended = "t0user_" + str(user.id)
            attributes_dict[user_prepended] = {
                'partition': 0,
                'age': str(user.age),
            }

        # Load all movie indices, but without attributes.
        movies = self._load_movie_data()
        for movie in movies:
            movie_prepended = "t1movie_" + str(movie)
            attributes_dict[movie_prepended] = {'partition': 1}

        attributes = pd.DataFrame.from_dict(attributes_dict, orient='index')
        return attributes

    def _load_ml_file(self, file_name, delimiter):
        return super()._load_file(file_name, delimiter, encoding='ISO-8859-1')

    def _load_user_movie_data(self):
        return self._load_ml_file("u.data", "\t").astype(np.int)

    def _load_user_data(self):
        user_data = self._load_ml_file("u.user", "|")
        users = []
        for data_row in user_data:
            users.append(User(data_row))
        return users

    def _load_movie_data(self):
        movie_data = self._load_ml_file("u.item", "|")
        movies = []
        for data_row in movie_data:
            movies.append(int(data_row[0]))
        return movies


class User:
    def __init__(self, data_row):
        self.id = int(data_row[0])

        age = int(data_row[1])
        age_brackets = [1, 18, 25, 35, 45, 50, 56, 1000]
        for i in range(1, len(age_brackets)):
            if age < age_brackets[i]:
                self.age = age_brackets[i - 1]
                break

        self.gender = data_row[2]
        self.occupation = data_row[3]
        self.zip = data_row[4]

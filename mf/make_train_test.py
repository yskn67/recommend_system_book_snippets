#! /usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../ml-latest-small/ratings.csv')
train, test = train_test_split(df, test_size=0.2, random_state=57)
train[['userId', 'movieId', 'rating']].to_csv('train.csv', header=False, index=False)
test[['userId', 'movieId', 'rating']].to_csv('test.csv', header=False, index=False)

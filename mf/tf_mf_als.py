#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib


class MatrixFactorizationALS(object):

    def __init__(self, n_users, n_items, factor_dim=16, reg_param=0.1, *args, **kwargs):
        self.n_users = n_users
        self.n_items = n_items
        self.factor_dim = factor_dim
        self.user_factor = tf.random.normal((n_users, factor_dim), mean=3, stddev=1, dtype=tf.float32)
        self.item_factor = tf.random.normal((n_items, factor_dim), mean=3, stddev=1, dtype=tf.float32)
        self.reg_param = reg_param

    def call(self, user_idx, item_idx):
        user_vec = tf.nn.embedding_lookup(self.user_factor, user_idx)
        item_vec = tf.nn.embedding_lookup(self.item_factor, item_idx)
        return tf.reduce_sum(tf.multiply(item_vec, user_vec), 1)

    def set_dataset(self, dataset):
        numpy_train_dataset = np.zeros((self.n_users, self.n_items))
        for user_idx, item_idx, ratings in dataset:
            for uidx, iidx, r in zip(user_idx, item_idx, ratings):
                numpy_train_dataset[uidx, iidx] = r
        self.train_dataset = tf.Variable(numpy_train_dataset, dtype=tf.float32)

    def loss(self, user_idx, item_idx, ratings):
        user_vec = tf.nn.embedding_lookup(self.user_factor, user_idx)
        item_vec = tf.nn.embedding_lookup(self.item_factor, item_idx)
        preds = tf.reduce_sum(tf.multiply(item_vec, user_vec), 1)
        loss = tf.square(tf.subtract(ratings, preds))
        user_reg = tf.multiply(self.reg_param, tf.sqrt(tf.reduce_sum(tf.square(user_vec), 1)))
        item_reg = tf.multiply(self.reg_param, tf.sqrt(tf.reduce_sum(tf.square(item_vec), 1)))
        return tf.add(loss, user_reg, item_reg)

    def _update_user_factor(self):
        term1 = tf.matmul(self.train_dataset, self.item_factor)
        term2 = tf.matmul(self.item_factor, self.item_factor, transpose_a=True)
        term3 = tf.linalg.diag(
            tf.constant(self.reg_param, dtype=tf.float32, shape=(self.factor_dim,)))
        # return tf.matmul(term1, tf.linalg.inv(tf.add(term2, term3)))
        return tf.transpose(tf.linalg.solve(tf.add(term2, term3), tf.transpose(term1)))

    def _update_item_factor(self):
        term1 = tf.matmul(tf.transpose(self.train_dataset), self.user_factor)
        term2 = tf.matmul(self.user_factor, self.user_factor, transpose_a=True)
        term3 = tf.linalg.diag(
            tf.constant(self.reg_param, dtype=tf.float32, shape=(self.factor_dim,)))
        # return tf.matmul(term1, tf.linalg.inv(tf.add(term2, term3)))
        return tf.transpose(tf.linalg.solve(tf.add(term2, term3), tf.transpose(term1)))

    def update_factor(self):
        self.user_factor = self._update_user_factor()
        self.item_factor = self._update_item_factor()


train_dataset = tf.data.experimental.CsvDataset(
    ['train.csv'],
    [tf.int32, tf.int32, tf.float32],
    header=False).batch(128)
test_dataset = tf.data.experimental.CsvDataset(
    ['test.csv'],
    [tf.int32, tf.int32, tf.float32],
    header=False).batch(128)
mf = MatrixFactorizationALS(611, 193610)
mf.set_dataset(train_dataset)

train_loss = []
test_loss = []
for epoch in range(50):
    print(f'epoch: {epoch}')
    mf.update_factor()
    epoch_train_loss = []
    for user_idx, item_idx, ratings in train_dataset:
        loss = mf.loss(user_idx, item_idx, ratings)
        epoch_train_loss.append(loss.numpy())

    epoch_train_loss = np.mean([li for l in epoch_train_loss for li in l])
    print(f'train loss: {epoch_train_loss}')
    train_loss.append(epoch_train_loss)

    epoch_test_loss = []
    for user_idx, item_idx, ratings in test_dataset:
        loss = mf.loss(user_idx, item_idx, ratings)
        epoch_test_loss.append(loss.numpy())

    epoch_test_loss = np.mean([li for l in epoch_test_loss for li in l])
    print(f'test loss: {epoch_test_loss}')
    test_loss.append(epoch_test_loss)

joblib.dump(train_loss, 'mf_als_train_loss.pkl')
joblib.dump(test_loss, 'mf_als_test_loss.pkl')

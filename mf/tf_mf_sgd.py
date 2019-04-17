#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib


class MatrixFactorizationSGD(tf.keras.Model):

    def __init__(self, n_users, n_items, factor_dim=16, reg_param=0.1, *args, **kwargs):
        super(MatrixFactorizationSGD, self).__init__(*args, **kwargs)
        self.reg_param = reg_param
        self.user_factor = tf.keras.layers.Embedding(n_users, factor_dim, input_length=1)
        self.item_factor = tf.keras.layers.Embedding(n_items, factor_dim, input_length=1)

    def call(self, user_idx, item_idx):
        user_vec = self.user_factor(user_idx)
        item_vec = self.item_factor(item_idx)
        return tf.reduce_sum(tf.multiply(item_vec, user_vec), 1)

    def loss(self, user_idx, item_idx, ratings):
        loss = tf.square(tf.subtract(ratings, self.call(user_idx, item_idx)))
        user_reg = tf.multiply(self.reg_param, tf.sqrt(tf.reduce_sum(tf.square(self.user_factor(user_idx)), 1)))
        item_reg = tf.multiply(self.reg_param, tf.sqrt(tf.reduce_sum(tf.square(self.item_factor(item_idx)), 1)))
        return tf.add(loss, user_reg, item_reg)


train_dataset = tf.data.experimental.CsvDataset(
    ['train.csv'],
    [tf.int32, tf.int32, tf.float32],
    header=False).shuffle(128).batch(128)
test_dataset = tf.data.experimental.CsvDataset(
    ['test.csv'],
    [tf.int32, tf.int32, tf.float32],
    header=False).shuffle(128).batch(128)
mf = MatrixFactorizationSGD(611, 193610)
optimizer = tf.keras.optimizers.SGD(lr=0.01)

train_loss = []
test_loss = []
for epoch in range(50):
    print(f'epoch: {epoch}')
    epoch_train_loss = []
    for user_idx, item_idx, ratings in train_dataset:
        with tf.GradientTape() as tape:
            loss = mf.loss(user_idx, item_idx, ratings)

        epoch_train_loss.append(loss.numpy())
        grads = tape.gradient(loss, mf.trainable_variables)
        optimizer.apply_gradients(zip(grads, mf.trainable_variables))
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

joblib.dump(train_loss, 'mf_sgd_train_loss.pkl')
joblib.dump(test_loss, 'mf_sgd_test_loss.pkl')

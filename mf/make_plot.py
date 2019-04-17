#! /usr/bin/env python

import matplotlib.pyplot as plt
from sklearn.externals import joblib


sgd_train_loss = joblib.load('mf_sgd_train_loss.pkl')
sgd_test_loss = joblib.load('mf_sgd_test_loss.pkl')
als_train_loss = joblib.load('mf_als_train_loss.pkl')
als_test_loss = joblib.load('mf_als_test_loss.pkl')
epochs = list(range(1, 51))


plt.figure()
plt.plot(epochs, sgd_train_loss, 'k--', label='Train')
plt.plot(epochs, sgd_test_loss, 'r-', label='Test')
plt.title('SGD Loss')
plt.xlabel('Epochs')
plt.ylabel('SGD Loss')
plt.legend(loc='upper right')
plt.savefig('sgd_loss.png')

plt.figure()
plt.plot(epochs, als_train_loss, 'k--', label='Train')
plt.plot(epochs, als_test_loss, 'r-', label='Test')
plt.title('ALS Loss')
plt.xlabel('Epochs')
plt.ylabel('ALS Loss')
plt.legend(loc='upper right')
plt.savefig('als_loss.png')

plt.figure()
plt.plot(epochs, sgd_train_loss, 'k--', label='SGD')
plt.plot(epochs, als_train_loss, 'r-', label='ALS')
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend(loc='upper right')
plt.savefig('train_loss.png')

plt.figure()
plt.plot(epochs, sgd_test_loss, 'k--', label='SGD')
plt.plot(epochs, als_test_loss, 'r-', label='ALS')
plt.title('Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend(loc='upper right')
plt.savefig('test_loss.png')

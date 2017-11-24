from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.RandomAction
from lib.env.mountain_car import MountainCarEnv
import matplotlib.pyplot as plt
import os
import lib.qlearning as ql
import pickle


# Create model
def neural_net(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def one_step_transition_model(learning_rate=0.1, n_hidden_1 = 32, n_hidden_2 = 32, num_input = 5, num_output = 4):

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }

    # Construct model
    logits = neural_net(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.losses.mean_squared_error(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    return loss_op, train_op

def get_train_test_data():
    mc2d_env = MountainCarEnv()
    mc3d_env = ThreeDMountainCarEnv()
    



def calculate_mapping():
    pass


if __name__ == 'main':
    calculate_mapping()
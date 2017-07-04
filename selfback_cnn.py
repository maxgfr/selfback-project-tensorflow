'''
A Convolutional Network for HAR based on : rgu-selfback

Author: Maxime Golfier
'''

from __future__ import print_function
from math import *

import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import csv

####PART FOR DATA####

def create_data(directory):
    list = []
    column_names = ['x-axis', 'y-axis', 'z-axis','activity']
    for i in range(1,6):
        name = directory+str(i)+'.csv'
        data = pd.read_csv(name, header=1, names=column_names)
        list.append(data)
    print(list)
    print('create_data is done')
    return list

def read_csv(numlines,path):

    filename_queue = tf.train.string_input_producer([path])

    reader = tf.TextLineReader(skip_header_lines=1)

    key, value = reader.read(filename_queue)

    record_defaults = [tf.constant([0], dtype=tf.float32),    # Column 1
                       tf.constant([0], dtype=tf.float32),    # Column 2
                       tf.constant([0], dtype=tf.float32),    # Column 3
                       tf.constant([], dtype=tf.int32)]  # Column 4

    col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)

    features = tf.stack([col1, col2, col3])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(numlines):
            # Retrieve a single instance:
            example, label = sess.run([features, col4])
            #print(example, label)

    coord.request_stop()
    coord.join(threads)

def count_lines(path):
    with open(path,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    return row_count

def read_one_csv(name):
    data = np.genfromtxt(name, delimiter=',', skip_header=1)
    return data

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def read_all_csv(directory,X,Y):
    all_data = []
    all_label = []
    all_lines_per_file = []
    for i in range(1,6):
        name = directory+str(i)+'.csv'
        data = read_one_csv(name)
        label = one_hot_encoded(i-1,6)
        line = count_lines(name)
        all_data.append(data)
        all_label.append(label)
        all_lines_per_file.append(line)
    #stuff = [[all_data[j], all_label[j]] for j in range(len(all_data))]
    x, y = format_data_label(all_data,all_label,all_lines_per_file)
    stuff = {X: x, Y: y}
    print('read_all_csv is done :)')
    return stuff

def format_data_label(all_data, all_label,all_line):
    x = []
    y = []
    for i in range(5):
        nb_sections = all_line[i]/500
        nb_sections = int(floor(nb_sections))
        nb_elmts_to_delete = all_line[i]%500
        tab = all_data[i][:nb_elmts_to_delete-1]
        newSection = np.array_split(tab, nb_sections)
        for j in range(nb_sections):
            label = copy.copy(all_label[i])
            x.append(newSection)
            y.append(label)

    return x, y


def placeholder_input(num_input, num_class):
    x = tf.placeholder(tf.float32, [500, num_input])
    y = tf.placeholder(tf.float32, [None, num_class])
    return x,y

####PART FOR MODEL####
def create_net(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

##################MAIN##################

# Parameters
learning_rate = 0.01
training_iters = 250
batch_size = 500
training_epochs = 100
n_input = 3 # data input
n_classes = 6 # data total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units
n_hidden_1 = 20
n_hidden_2 = 20
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


x, y = placeholder_input(n_input,n_classes)
feed_dict = read_all_csv('data_light/', x, y)
#print(feed_dict)

# Construct model
pred = create_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    # Training cycle
    for step in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict=feed_dict)

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


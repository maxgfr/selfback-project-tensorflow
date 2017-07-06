'''
A Convolutional Network for HAR based on : rgu-selfback

Author: Maxime Golfier
'''

from __future__ import print_function
from math import *
from scipy import stats
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

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
    return int(row_count)

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
    for i in range(1,7):
        name = directory+str(i)+'.csv'
        data = read_one_csv(name)
        #data = data.astype(np.float32)
        label = one_hot_encoded(i-1,6)
        line = count_lines(name)
        all_data.append(data)
        all_label.append(label)
        all_lines_per_file.append(line)

    #print(all_data)
    #print(all_label)[
    #stuff = [[all_dataj], all_label[j]] for j in range(len(al# l_data))]
    res1, res2 = format_data_label(all_data,all_label,all_lines_per_file)
    print('read_all_csv is done :)')
    return res1, res2

def input_pipeline(batch_size, example_list):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

def format_data_label(all_data, all_label,all_line):
    res1 = []
    res2 = []
    for i in range(6):
        nb_sections = all_line[i]/500
        nb_sections = int(floor(nb_sections))
        nb_elmts_to_delete = all_line[i]%500
        tab = all_data[i][:nb_elmts_to_delete-1]
        newSection = np.array_split(tab, nb_sections)
        for j in range(nb_sections):
            label = copy.copy(all_label[i])
            res1.append(newSection)
            res2.append(500*label)
    return res1, res2

def read_data(file_path):
    column_names = ['x-axis', 'y-axis', 'z-axis', 'activity']
    data = pd.read_csv(file_path, header=1, names=column_names)
    return data

def create_datasets (data, lines,length=500):
    ds = np.empty((0,length,3))
    labels = np.empty((0))
    nb_sections = lines/500
    nb_sections = int(floor(nb_sections))
    for i in range(nb_sections):
        stop = length*(i+1)
        start = stop - 500
        x = data["x-axis"][start:stop]
        y = data["y-axis"][start:stop]
        z = data["z-axis"][start:stop]

        ds = np.vstack([ds, np.dstack([x, y, z])])
        labels = np.append(labels, stats.mode(data["activity"][start:stop])[0][0])
    return ds, labels

####PART FOR MODEL####
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, keep_prob):
    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)

    # Convolution Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    # Convolution Layer 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3)

    # Convolution Layer 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4)

    # Convolution Layer 5
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5)

    # Fully connected layer 1
    # Reshape conv5 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.layers.dense(inputs=fc1, units=900, activation=tf.nn.tanh)

    # Fully connected layer 2
    dense2 = tf.layers.dense(dense1, units=300, activation=tf.nn.tanh)
    dropout = tf.layers.dropout(inputs=dense2, rate=0.5)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=6)

    # Output, class prediction
    out = tf.nn.softmax(logits, name='output')
    return out

def placeholder_input(input_height, input_width, num_channels,  num_class):
    x = tf.placeholder(tf.float32, [None, input_height, input_width, num_channels], name='input')
    y = tf.placeholder(tf.float32, [None, num_class], name='output')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return x, y, keep_prob

def export_model(input_node_names, output_node_name , model_name):
    freeze_graph.freeze_graph('out/' + model_name + '.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph saved!")

##################MAIN##################
# Parameters
file = 'data/allDataLight.csv'
model_name = 'cnn_wrist500_tf'
training_epochs = 10
learning_rate = 0.01
n_input = 3
n_height = 1
n_width = 500
n_channels = 3
n_classes = 6
batch_size = 500
weights = {
    # 1x10 conv, 3 input, 150 outputs
    'wc1': tf.Variable(tf.random_normal([1, 10, 3, 150])),
    # 1x10 conv, 150 input, 100 outputs
    'wc2': tf.Variable(tf.random_normal([1, 10, 150, 100])),
    # 1x10 conv, 100 input, 80 outputs
    'wc3': tf.Variable(tf.random_normal([1, 10, 100, 80])),
    # 1x10 conv, 80 input, 60 outputs
    'wc4': tf.Variable(tf.random_normal([1, 10, 80, 60])),
    # 1x10 conv, 60 input, 40 outputs
    'wc5': tf.Variable(tf.random_normal([1, 10, 60, 40])),
    # fully connected, 500/2^5 => 15.625 inputs, 900 outputs
    'wd1': tf.Variable(tf.random_normal([16*1*40, 900]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([150])),
    'bc2': tf.Variable(tf.random_normal([100])),
    'bc3': tf.Variable(tf.random_normal([80])),
    'bc4': tf.Variable(tf.random_normal([60])),
    'bc5': tf.Variable(tf.random_normal([40]))
}
X, Y, keep_prob = placeholder_input(n_height, n_width, n_channels, n_classes)

#Read Data
dataset = read_data(file)
numlines = count_lines(file)

#Create datasets from the file
data, labels = create_datasets(dataset, numlines)

#Reshape data
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
reshaped_data = data.reshape(len(data), n_height, n_width, n_channels)

#Split data to test it
train_test_split = np.random.rand(len(reshaped_data)) < 0.85

#Create train et test data
train_x = reshaped_data[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_data[~train_test_split]
test_y = labels[~train_test_split]

# Construct model
pred = conv_net(X, weights, biases, keep_prob)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    tf.train.write_graph(sess.graph_def, 'out',
                         model_name + '.pbtxt', True)

    # Keep training until reach max iterations
    for step in range(training_epochs):

        offset = (step * batch_size) % (train_y.shape[0] - batch_size)
        batch_data = train_x[offset:(offset + batch_size), :]
        batch_labels = train_y[offset:(offset + batch_size)]

        #make evaluation of the accuracy each 5 epochs
        if step % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X: batch_data, Y: batch_labels, keep_prob: 1.0})
            print('step %d, training accuracy %f' % (step, train_accuracy))

        _, summary = sess.run([optimizer, loss], feed_dict={X: batch_data, Y: batch_labels, keep_prob: 0.5})

        print(str(step), ' epoch(s) completed')

    saver.save(sess, 'out/' + model_name + '.chkp')

    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y, keep_prob: 1.0}))

    export_model(["input", "keep_prob"], "output", model_name)




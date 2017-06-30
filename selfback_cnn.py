'''
A Convolutional Network for HAR based on : rgu-selfback

Author: Maxime Golfier
'''

from __future__ import print_function

import tensorflow as tf
import csv

#PART FOR DATA
def count_lines(path):
    with open(path,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    return row_count

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

def one_hot_encoding():
    all_activities = [0,1,2,3,4,5]
    onehot = {}
    # Target number of activities types (target classes) is 3 ^
    activities_count = len(all_activities)

    # Print out each one-hot encoded string for 6 activities.
    for i, activities in enumerate(all_activities):
        # %0*d gives us the second parameter's number of spaces as padding.
        print("%s,%0*d" % (activities, activities_count, 10 ** i))

# PART FOR TUNING
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# PART FOR CREATION OF MODEL
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



##################MAIN##################
data = "data/allDataUltraLight.csv"
numlines = count_lines(data)
print(numlines," is the number of line on this csv")
read_csv(numlines,data)
print("read_csv is done")


# Parameters
learning_rate = 0.01
training_iters = 250
batch_size = 500
display_step = 10

# Network Parameters
n_input = 500*3*1 # data input
n_classes = 9 # data total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()




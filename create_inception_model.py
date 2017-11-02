import tensorflow as tf
import numpy as np
import os
import PIL.Image
import glob
import random
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
! Right now this code probably doesn't work as intended yet !
"""


export_dir = os.getcwd() + '/saved_models/'

batch_size = 2
image_dim = 100
num_feature_maps = 10
num_reduce = 16
fc1_size = 100
fc2_size = 2
dropout = 0.5


# load the dataset

# change this to the correct path
img_folder = '/home/odin/Desktop/div/random.jpg'

labels = []

# create float arrays from the input images
cat_images = []
for filename in glob.glob('/home/odin/Desktop/data/cat/*.jpg'):
    img = np.float32(PIL.Image.open(filename))
    cat_images.append(img)
    labels.append(0)

dog_images = []
for filename in glob.glob('/home/odin/Desktop/data/dog/*.jpg'):
    img = np.float32(PIL.Image.open(filename))
    dog_images.append(img)
    labels.append(1)

# add them together
images = np.concatenate((cat_images, dog_images))

# randomize the input
temp = list(zip(images, labels))
random.shuffle(temp)
images, labels = zip(*temp)

# create one hot vectors from labels
target = tf.one_hot(labels, depth=2)
target = tf.Session().run(target)


# get the accuracy
def accuracy(target, predictions):
    return 100.0*np.sum(np.argmax(target,1) == np.argmax(predictions,1))/target.shape[0]


graph = tf.Graph()
with graph.as_default():
    # training data + labels
    X = tf.placeholder(tf.float32, shape=(batch_size, image_dim, image_dim, 3))
    y_ = tf.placeholder(tf.float32, shape=(batch_size, 2))

    def createWeight(size, Name):
        return tf.Variable(tf.truncated_normal(size, stddev=0.1), name=Name)

    def createBias(size, Name):
        return tf.Variable(tf.constant(0.1, shape=size), name=Name)

    def conv2d_s1(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_3x3_s1(x):
        return  tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')


    def add_inception_module(input, num_feature_maps, num_reduce, input_channels):

        # follows input
        W_conv_1x1_1 = createWeight([1, 1, input_channels, num_feature_maps], 'W_conv1_1x1_1')
        b_conv_1x1_1 = createBias([num_feature_maps], 'b_conv1_1x1_1')
        conv_1x1_1 = conv2d_s1(input, W_conv_1x1_1) + b_conv_1x1_1

        # follows input
        W_conv_1x1_2 = createWeight([1, 1, input_channels, num_reduce], 'W_conv1_1x1_2')
        b_conv_1x1_2 = createBias([num_reduce], 'b_conv1_1x1_2')
        conv_1x1_2 = tf.nn.relu(conv2d_s1(input, W_conv_1x1_2) + b_conv_1x1_2)
        # follows 1x1_2
        W_conv_3x3 = createWeight([3, 3, num_reduce, num_feature_maps], 'W_conv1_3x3')
        b_conv_3x3 = createBias([num_feature_maps], 'b_conv1_3x3')
        conv_3x3 = conv2d_s1(conv_1x1_2, W_conv_3x3) + b_conv_3x3

        # follows input
        W_conv_1x1_3 = createWeight([1, 1, input_channels, num_reduce], 'W_conv1_1x1_3')
        b_conv_1x1_3 = createBias([num_reduce], 'b_conv1_1x1_3')
        conv_1x1_3 = tf.nn.relu(conv2d_s1(input, W_conv_1x1_3) + b_conv_1x1_3)
        # follows 1x1_3
        W_conv_5x5 = createWeight([5, 5, num_reduce, num_feature_maps], 'W_conv1_5x5')
        b_conv_5x5 = createBias([num_feature_maps], 'b_conv1_5x5')
        conv_5x5 = conv2d_s1(conv_1x1_3, W_conv_5x5) + b_conv_5x5

        maxpool = max_pool_3x3_s1(input)
        # follows max pooling
        W_conv_1x1_4 = createWeight([1, 1, input_channels, num_feature_maps], 'W_conv1_1x1_4')
        b_conv_1x1_4 = createBias([num_feature_maps], 'b_conv1_1x1_4')
        conv_1x1_4 = conv2d_s1(maxpool, W_conv_1x1_4) + b_conv_1x1_4


        # concatenate all the feature maps and hit them with a relu
        inception = tf.nn.relu(tf.concat([conv_1x1_1, conv_3x3, conv_5x5, conv_1x1_4], 3))

        return inception

    def model(input, train=True):

        inception1 = add_inception_module(input, num_feature_maps, num_reduce, 3)
        inception2 = add_inception_module(inception1, num_feature_maps, num_reduce, int(inception1.shape[3]))

        inception2_flat = tf.reshape(inception2, [-1, image_dim*image_dim*4*num_feature_maps])

        # fully connected layers
        W_fc1 = createWeight([image_dim*image_dim*(4*num_feature_maps), fc1_size], 'W_fc1')
        b_fc1 = createBias([fc1_size], 'b_fc1')
        if train:
            h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), dropout)
        else:
            h_fc1 = tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1)

        W_fc2 = createWeight([fc1_size, fc2_size], 'W_fc2')
        b_fc2 = createBias([fc2_size], 'b_fc2')
        return tf.matmul(h_fc1, W_fc2)+b_fc2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model(X), labels=y_))
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

    init = tf.global_variables_initializer()

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

num_steps = 100
sess = tf.Session(graph=graph)

sess.run(init)
print("initialized model")

for i in range(num_steps):
    offset = (i * batch_size) % (len(images) - batch_size)
    batch_x, batch_y = images[offset:(offset + batch_size)], target[offset:(offset + batch_size)]
    feed_dict = {X: batch_x, y_: batch_y}
    _, loss_value = sess.run([opt, loss], feed_dict=feed_dict)
    print("step:", i, "     loss:", loss_value)

    if i == (num_steps-1):
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING],
                                             signature_def_map=None,
                                             assets_collection=None)

builder.save()



import os
import tensorflow as tf
import numpy as np
import random
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# change this to the correct path
img_path = '/home/odin/Desktop/div/random.jpg'

# create a float array from the input image
img = np.float32(PIL.Image.open(img_path))


# create and show image from float array
def show_arrayimg(array):
    array = np.clip(array / 255.0, 0, 1) * 255
    dreamed_image = PIL.Image.fromarray(array.astype('uint8'))
    dreamed_image.show()

# show_arrayimg(img)


# load the model
def load_model(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# change this to the correct path
model_path = 'models/tensorflow_inception_graph.pb'

load_model(model_path)

# start a tensorflow session
sess = tf.Session()


def print_layer_names():
    for op in sess.graph.get_operations():
        print(op.name)

print_layer_names()


# The following function splits the image into smaller segments (grid style), computes gradients for
# each segment, and concatenates the results into a gradient for the entire image. This gets rid of
# GPU-limitations, so we are free to use as large pictures as we want to.
def compute_gradient_from_image_segments(image, tensor, channel=None, segment_dim=200):

    # Squaring the tensor gives the feature detection in images higher discrimination?? (pictures look better)
    tensor = tf.square(tensor)

    # The mean is calculated from the chosen channels in the layer. If None, we use the mean from the entire layer.
    if channel is None:
        tensor_mean = tf.reduce_mean(tensor[:,:,:,:])
    else:
        tensor_mean = tf.reduce_mean(tensor[:,:,:,channel])

    # get the input tensor
    input_tensor = sess.graph.get_tensor_by_name("input:0")

    # Tensorflow automatically generates a function for computing the gradient
    gradient_function = tf.gradients(tensor_mean, input_tensor)[0]

    # array containing zeroes, which will be filled up with gradients from each segment
    gradient = np.zeros_like(image)
    x_max = len(gradient)
    y_max = len(gradient[0])

    # to divide the picture differently each time, we introduce some randomness
    random_offset = int(segment_dim / 2)

    # the starting point of the grid we use to segment the image
    x_start = random.randint(-random_offset, 0)
    while x_start < x_max:

        # find the beginning and end of the current segment. Can't be outside the image
        x_start = max(0, x_start)
        x_end = min(x_start + segment_dim, x_max)

        # randomness in the y-direction
        y_start = random.randint(-random_offset, 0)
        while y_start < y_max:

            y_start = max(0, y_start)
            y_end = min(y_start + segment_dim, y_max)

            # get the current image-segment, witch we will compute the gradient for
            image_segment = image[x_start:x_end, y_start:y_end, :]
            # we must add an extra dimension, because the inception-network can take in multiple images at the same time
            image_segment = np.expand_dims(image_segment, axis=0)

            feed_dict = {"input:0": image_segment}

            # compute the gradient for the current segment
            segment_gradient = sess.run(gradient_function, feed_dict=feed_dict)

            # normalizing the gradient
            segment_gradient = segment_gradient/(1e-8 + np.std(segment_gradient))

            # adding the segment-gradient to the gradient for the entire image
            gradient[x_start:x_end, y_start:y_end, :] = segment_gradient
            y_start = y_end

        # continue the next segment, where the last one ended
        x_start = x_end

    return gradient


# the deepdream algorithm, which alters the input-image according to the gradient computed in every iteration
def deepdream(image, layer_name, iterations=30, step_size=3, channel=None):

    # get the tensor we will use to compute the gradient
    tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
    print(tensor)  # when printing the tensor, we can see how many channels it contains

    # make a copy of the original image
    dreamed_image = image.copy()

    # altering the copied image a little bit every iteration
    for i in range(iterations):
        print("iteration: ", i)

        # compute the gradient for the entire image
        gradient = compute_gradient_from_image_segments(dreamed_image, tensor, channel)

        # add the gradient to the image
        dreamed_image += gradient * step_size

        # show the image during the run
        if i % 10 == 0:
            show_arrayimg(dreamed_image)

    show_arrayimg(dreamed_image)
    return dreamed_image


# The following function runs the deepdream-algorithm on different scales (octaves) of the original image.
# This is done to discover patterns of different sizes.
def deepdream_with_octaves(image, layer_name, iterations=100, step_size=3, channel=None, octaves=4, scale_ratio=0.8, octave_blend=0.7):
    image_height = len(image)
    image_with = len(image[0])

    # converts the image-array back to an image object for easy manipulation
    image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')

    # iterates through different octaves, starting with the smallest
    for i in range(octaves-1, -1, -1):

        # scale image
        new_height = int(image_height * scale_ratio**i)
        new_with = int(image_with * scale_ratio**i)
        scaled_image = image.resize((new_with, new_height))

        # convert image to array
        scaled_image_array = np.float32(scaled_image)

        # run deepdream algorithm
        dreamed_image_array = deepdream(scaled_image_array, layer_name, iterations, step_size, channel)

        # create image object of the image-array
        dreamed_image_array = np.clip(dreamed_image_array / 255.0, 0, 1) * 255
        dreamed_image = PIL.Image.fromarray(dreamed_image_array.astype('uint8'))
        dreamed_image.show()

        # resize back to the original size
        dreamed_image = dreamed_image.resize((image_with, image_height), PIL.Image.BICUBIC)

        # blend the previous high resolution image with the dreamed image
        image = PIL.Image.blend(image, dreamed_image, octave_blend)
        image.show()

    return np.float32(image)


deepdream(img, 'mixed4c', iterations=300)
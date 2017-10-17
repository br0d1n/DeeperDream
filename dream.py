import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# change this to the correct path
img_path = '/home/odin/Desktop/div/cat.jpg'

# create a float array from the input image
img = np.float32(PIL.Image.open(img_path))


# create and show image from float array
def show_arrayimg(array):
    image = np.uint8(np.clip(array/255.0, 0, 1) * 255)
    plt.imshow(image)
    plt.show()

# show_arrayimg(img)


# load the model
def load_model(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


model_path = 'models/tensorflow_inception_graph.pb'

load_model(model_path)

# start a tensorflow session
sess = tf.Session()


def print_layer_names():
    for op in sess.graph.get_operations():
        print(op.name)

print_layer_names()


# Computes the gradient for an entire image. Must me small enough to not exceed GPU-limitations
# could probably be DEPRECATED
def compute_gradient(image, tensor):
    tensor = tf.square(tensor) # why square the tensor?
    tensor_mean = tf.reduce_mean(tensor) # why dis?
    input_tensor = sess.graph.get_tensor_by_name("input:0")
    gradient_function = tf.gradients(tensor_mean, input_tensor)[0]

    image = np.expand_dims(image, axis=0)

    gradient = sess.run(gradient_function, feed_dict={"input:0": image})
    gradient = gradient/(np.std(gradient) + 1e-8) # why normalize?
    return gradient


# The following function splits the image into smaller segments (grid style), computes gradients for
# each segment, and concatenates the results into a gradient for the entire image. This gets rid of
# GPU-limitations, so we are free to use as large pictures as we want to.
def compute_gradient_from_image_segments(image, tensor, segment_dim=200):
    tensor = tf.square(tensor)
    tensor_mean = tf.reduce_mean(tensor)
    input_tensor = sess.graph.get_tensor_by_name("input:0")
    gradient_function = tf.gradients(tensor_mean, input_tensor)[0]

    # array containing zeroes, which will be filled up with gradients from each segment
    gradient = np.zeros_like(image)
    x_max = len(gradient)
    y_max = len(gradient[0])

    # to divide the picture differently each time, we introduce some randomness
    random_offset = segment_dim / 4

    # the starting point of the grid we use to segment the image
    x_start = random.randint(-random_offset, 0)
    while x_start < x_max:
        x_end = min(x_start + segment_dim, x_max)
        x_start = max(0, x_start)

        y_start = random.randint(-random_offset, 0)
        while y_start < y_max:
            y_end = min(y_start + segment_dim, y_max)
            y_start = max(0, y_start)

            image_segment = image[x_start:x_end, y_start:y_end, :]
            image_segment = np.expand_dims(image_segment, axis=0)

            feed_dict = {"input:0": image_segment}

            segment_gradient = sess.run(gradient_function, feed_dict=feed_dict)

            # normalizing the gradient
            segment_gradient = segment_gradient/(np.std(segment_gradient)+1e-8)

            # adding the segment-gradient to the gradient for the entire image
            gradient[x_start:x_end, y_start:y_end, :] = segment_gradient
            y_start = y_end

        x_start = x_end

    return gradient


def deepdream(image, layer_name, iterations=300, step_size=2):
    tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
    compute_gradient_from_image_segments(image, tensor)
    dreamed_image = image.copy()
    for i in range(iterations):
        print("iteration: ", i)
        gradient = compute_gradient_from_image_segments(dreamed_image, tensor)
        dreamed_image += gradient * step_size
        #if i % 20 == 0:
        #    show_arrayimg(dreamed_image)
    #show_arrayimg(dreamed_image)
    return dreamed_image


# The following function runs the deepdream-algorithm on different scales (octaves) of the original image.
# This is done to discover patterns of different sizes.
def deepdream_with_octaves(image, layer_name, iterations=100, step_size=2, octaves=4, scale_ratio=0.8):
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
        dreamed_image_array = deepdream(scaled_image_array, layer_name, iterations)

        # create image object of the image-array
        dreamed_image_array = np.clip(dreamed_image_array / 255.0, 0, 1) * 255
        dreamed_image = PIL.Image.fromarray(dreamed_image_array.astype('uint8'))
        dreamed_image.show()

        # resize back to the original size
        dreamed_image = dreamed_image.resize((image_with, image_height))

        # blend the previous high resolution image with the dreamed image
        image = PIL.Image.blend(image, dreamed_image, 0.8)

    return np.float32(image)



deepdream_with_octaves(img, 'mixed4c', iterations=100, octaves=6)

# deepdream(img, 'mixed3b')


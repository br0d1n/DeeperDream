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


def deepdream(image, layer_name, iterations=200, step_size=2):
    tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
    compute_gradient_from_image_segments(image, tensor)
    dreamed_image = image.copy()
    for i in range(iterations):
        print("iteration: ", i)
        gradient = compute_gradient_from_image_segments(dreamed_image, tensor)
        dreamed_image += gradient * step_size
        if i % 20 == 0:
            show_arrayimg(dreamed_image)
    show_arrayimg(dreamed_image)

deepdream(img, 'mixed5b_3x3')


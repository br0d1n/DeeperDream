import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# change this to the correct path
img_path = '/home/odin/Desktop/div/odin.jpg'

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

# print_layer_names()


def compute_gradient(image, tensor):

    tensor = tf.square(tensor) # why square the tensor?
    tensor_mean = tf.reduce_mean(tensor) # why dis?
    input_tensor = sess.graph.get_tensor_by_name("input:0")
    gradient_function = tf.gradients(tensor_mean, input_tensor)[0]

    # gradient = np.zeros_like(image)
    # print("x dim: ", len(gradient[0]))
    # print("y dim: ", len(gradient))

    image = np.expand_dims(image, axis=0)

    gradient = sess.run(gradient_function, feed_dict={"input:0": image})
    gradient = gradient/(np.std(gradient) + 1e-8) # why normalize?
    return gradient


def deepdream(image, layer_name, iterations=50, step_size=5):
    tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
    dreamed_image = image.copy()
    for i in range(iterations):
        print("iteration: ", i)
        gradient = compute_gradient(dreamed_image, tensor)[0]
        dreamed_image += gradient * step_size
        if i % 10 == 0:
            show_arrayimg(dreamed_image)
    show_arrayimg(dreamed_image)

deepdream(img, 'mixed5b')


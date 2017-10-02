import os
import scipy.ndimage as nd
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
def show_arrayimg(a):
    a = np.uint8(np.clip(a/255.0, 0, 1) * 255)
    plt.imshow(a)
    plt.show()

show_arrayimg(img)

# load the model
def load_model(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


model_path = 'models/tensorflow_inception_graph.pb'

load_model(model_path)

sess = tf.Session()
print(sess.graph.get_operations())
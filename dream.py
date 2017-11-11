import os
import tensorflow as tf
import numpy as np
import random
import PIL.Image, PIL.ImageTk
from tkinter import *
import matplotlib.pyplot as plt
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# toggle the gui on/off
use_GUI = False

# change this to the correct path
img_path = '/home/ole/Pictures/5dogs.jpg'

# create a float array from the input image
img = np.float32(PIL.Image.open(img_path))

# generate an image with random noise
def random_noise_img(dim):
    array = np.zeros((dim, dim, 3))
    for x in range(dim):
        for y in range(dim):
            array[x][y][0] = float(random.randint(0, 255))
            array[x][y][1] = float(random.randint(0, 255))
            array[x][y][2] = float(random.randint(0, 255))
    return array


# generate a completely grey image
def grey_img(dim):
    array = np.full((dim, dim, 3), 120)
    return array

#img = random_noise_img(500)


# create and show image from float array
def show_arrayimg(array):
    array = np.clip(array / 255.0, 0, 1) * 255
    dreamed_image = PIL.Image.fromarray(array.astype('uint8'))
    if use_GUI:
        global image
        image = PIL.ImageTk.PhotoImage(dreamed_image)
        canvas.itemconfig(imageSprite, image=image)
        canvas.update()
    else:
        dreamed_image.show()


# load the model
def load_model(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_model_2(model_path):
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], model_path)


# change this to the correct path
model_path = 'models/tensorflow_inception_graph.pb'
# model_path = 'saved_models/'


# start a tensorflow session
sess = tf.Session()

load_model(model_path)


# print the names of all layers in the network
def print_layer_names():
    for op in sess.graph.get_operations():
        if op.type != 'Const':
            print(op.type, "\t\t", op.name)

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
        x_end = min(x_start + segment_dim, x_max)
        x_start = max(0, x_start)

        # randomness in the y-direction
        y_start = random.randint(-random_offset, 0)
        while y_start < y_max:

            y_end = min(y_start + segment_dim, y_max)
            y_start = max(0, y_start)

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
def deepdream(image, layer_name, iterations=100, step_size=3, channel=None):

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
        if i % 2 == 0:
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

# deepdream(img, 'mixed4d_3x3_bottleneck_pre_relu', iterations=300, channel=139)


# ----------------------- TKINTER ------------------------------------------

if use_GUI:

    root = Tk()
    root.title("deeper dream")

    menuFrame = Frame(root)
    menuFrame.pack(side=TOP)

    imageFrame = Frame(root, height=500, width=700)
    imageFrame.pack(side=BOTTOM)

    canvas = Canvas(imageFrame, width=len(img[0]), height=len(img))
    canvas.pack(side=LEFT, fill=BOTH)
    original_img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img.astype('uint8')))
    imageSprite = canvas.create_image(0, 0, image=original_img, anchor=NW)

    def onselect_layer(event):
        channelMenu.delete(0, END)
        selected = layerMenu.curselection()
        layer_name = layerMenu.get(selected[0])
        tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
        num_channels = tensor.get_shape().as_list()[-1]
        channels = range(0, num_channels)
        channelMenu.insert(END, "all channels")
        for ch in channels:
            channelMenu.insert(END, ch)
        channelMenu.select_set(0)

    scrollbar = Scrollbar(menuFrame)
    layerMenu = Listbox(menuFrame, yscrollcommand=scrollbar.set, width=30)
    layerMenu.pack(side=LEFT)
    for op in sess.graph.get_operations():
        if op.type != 'Const' and op.type != 'Placeholder':
            layerMenu.insert(END, op.name)
    layerMenu.bind("<ButtonRelease-1>", onselect_layer)
    scrollbar.pack(side=LEFT, fill=Y)
    scrollbar.config(command=layerMenu.yview)

    scrollbar2 = Scrollbar(menuFrame)
    channelMenu = Listbox(menuFrame, yscrollcommand=scrollbar2.set)
    channelMenu.pack(side=LEFT)
    scrollbar2.pack(side=LEFT, fill=Y)
    scrollbar2.config(command=channelMenu.yview)


    def run(event):
        layer_name = layerMenu.get(ACTIVE)
        channel_name = channelMenu.get(ACTIVE)
        iterations = int(iterationsEntry.get())
        if channel_name == "all channels":
            deepdream(img, layer_name, iterations=iterations)
        else:
            print(layer_name)
            print(channel_name)
            deepdream(img, layer_name, iterations=iterations, channel=int(channel_name))


    rightFrame = Frame(menuFrame)
    rightFrame.pack(side=RIGHT)

    iterationsFrame = Frame(rightFrame)
    iterationsFrame.pack(side=TOP)

    iterationsLabel = Label(iterationsFrame, text="iterations")
    iterationsLabel.pack(side=LEFT)

    iterationsEntry = Entry(iterationsFrame)
    iterationsEntry.pack(side=RIGHT)
    iterationsEntry.insert(END, "100")

    runButton = Button(rightFrame, text="Run DeepDream")
    runButton.bind("<Button-1>", run)
    runButton.pack(side=BOTTOM)

    root.mainloop()

def plotNNFilter(layer_name):
    #Get the tensor by name
    tensor = sess.graph.get_tensor_by_name(layer_name + ':0')
    #Run the tensor with the image as input
    units = sess.run(tensor,feed_dict={"input:0":[img]})
    filters = units[0,:,:,:]
    filter_size = units.shape[3]
    width = units.shape[1]
    height = units.shape[2]
    plt.figure(1, figsize=(20,20))
    n_columns = 13
    n_rows = 10

    sorted_filters = list()
    for i in range(filter_size):
        fi = filters[:,:,i]
        sorted_filters.append((np.mean(fi),i,fi))
    sorted_filters = sorted(sorted_filters, reverse=True, key=lambda tup: tup[0])

    for i in range(25):

        filter_tuple = sorted_filters[i]

        newImg = PIL.Image.open(img_path)
        mask = PIL.Image.fromarray(filter_tuple[2]/filter_tuple[2].max())
        mask = np.float32(mask.resize((img.shape[1], img.shape[0])))
        r,g,b = newImg.split()

        r = PIL.Image.fromarray(np.uint8(r*mask))
        g = PIL.Image.fromarray(np.uint8(g*mask))
        b = PIL.Image.fromarray(np.uint8(b*mask))

        print("Showing filter:", filter_tuple[1], "Score:", filter_tuple[0])

        newImg = PIL.Image.merge('RGB', (r,g,b))
        newImg.show()

    #Plot the first 100 filter-activations
    # means = np.zeros(filters)
    # for i in range(filters):
    #     means[i] = np.mean(units[0,:,:,i])
    # k = 0
    # for i in range(filters):
    #     if(means[i] > 3 and k < n_columns*n_rows):
    #         print("Plotting filter nr. ", i)
    #         plt.subplot(n_rows, n_columns, k+1)
    #         plt.title("Mean:" + str(np.mean(units[0,:,:,i])))
    #         plt.imshow(units[0,:,:,i], interpolation="none", cmap="gray")
    #         k += 1
    #plt.tight_layout(h_pad=5.0)
    #plt.show()

plotNNFilter("mixed4e")

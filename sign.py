import tensorflow as tf
import os
from skimage.io import imread_collection
from skimage import transform
from skimage.color import rgb2gray
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    # 按照文件夹名字集成
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = glob.glob(os.path.join(label_directory, "*.ppm"))
        image_collection = imread_collection(file_names)
        images.extend(image_collection)
        # print('====')
        # print([int(d)], len(image_collection), [int(d)] * len(image_collection))
        labels.extend([int(d)] * len(image_collection))
    return images, labels


ROOT_PATH = "./img"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
# print(labels, type(labels), len(labels))
# print('==== before')
# labels = plt.hist(labels, 62)
# print(labels, type(labels), len(labels))
# plt.show()

# ===
# Determine the (random) indexes of the images that you want to see
traffic_signs = [300, 2250, 3650, 4000]
# Fill out the subplots with the random images that you defined
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
#     plt.show()
#     # shape 是一个元组，用于表示图像数组的形状。在这个例子中，images[traffic_signs[i]].shape 的值为 (285, 285, 3)，
#     # 表示该图像数组有 285 行、285 列和 3 个通道。
#     # 这个图像数组是一个 RGB 彩色图像，其中的每个像素由 3 个值组成，分别表示红色、绿色和蓝色通道的亮度值。
#     # 因此，这个图像数组的形状是一个三维数组，第一个维度表示行数，第二个维度表示列数，第三个维度表示通道数。
#     print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
#                                                   images[traffic_signs[i]].min(),
#                                                   images[traffic_signs[i]].max()))

# Get the unique labels
# 因为是概览图即可，所以，同类的就取一个图用来展示
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1
# ====== resize图片
# 将图片做成28*28
images28 = [transform.resize(image, (28, 28)) for image in images]
# ====== 灰度处理
# Convert `images28` to an array
images28 = np.array(images28)
# Convert `images28` to grayscale
images28 = rgb2gray(images28)

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images28[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image, cmap='gray')
    
# Show the plot
plt.show()

# Initialize placeholders 
# x = tf.keras.Input(shape=(28, 28))
# y = tf.keras.Input(shape=(), dtype=tf.int32)

# # Flatten the input data
# images_flat = tf.contrib.layers.flatten(x)

# # Fully connected layer 
# logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
# # Define a loss function 
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
# # Define an optimizer 
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# # Convert logits to label indexes
# correct_pred = tf.argmax(logits, 1)

# # Define an accuracy metric
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", correct_pred)
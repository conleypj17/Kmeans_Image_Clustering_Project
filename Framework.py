from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
import random
from matplotlib.widgets import Slider, Button


# ImageCompressor class that holds images
class ImageCompressor:
    def __init__(self, og_image=None):
        self.original_image = og_image  # original image attribute that holds the original image before kmeans clustering
        self.kmeans_images = {}  # dictionary that holds the kmeans images based on the value of k
        self.last_k = 0  # tracks the last displayed k value for this image
        for i in range(11):  # the range from k=0 to k=10
            self.kmeans_images[i] = self.kmeans(num=i)  # filling in the dictionary based on k

    def kmeans(self, num=0):
        if num == 0:  # if num is equal to 0, return the original image without any kmeans clustering
            return self.original_image

        pixels = self.original_image.reshape(-1, 3)  # formatting the image into R, G, and B colors
        pixel_count = pixels.shape[0]  # stores the number of pixels in the image

        threshold = 0.1  # holds the error threshold for the kmeans clustering

        random.seed(5)  # fixed random
        centroids = pixels[random.sample(range(pixel_count), num)]  # initially setting random centroids

        clusters = np.zeros(pixel_count, dtype=int)  # initially setting the clusters to 0

        for x in range(100):  # iterates 100 times
            # distance calculation using d = sqrt((r1-r2)^2 + (g1 - g2)^2 + (b1 - b2)^2)
            distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(
                axis=2))  # this is equivalent to the distance formula above
            clusters = np.argmin(distances, axis=1)  # storing index of smallest distance, aka the closest centroid

            # recomputes the centroid of each cluster by averaging the R, G, and B values of the pixels
            new_centroids = np.array([pixels[clusters == j].mean(axis=0) for j in range(num)])

            # checking to see if the error is less than the threshold of 0.1
            if np.allclose(centroids, new_centroids, atol=threshold):
                break  # will break out of loop if error is less than 0.1
            centroids = new_centroids  # storing the new centroids in centroid

        # assigns the pixels to the colors of their closest centroid
        compressed_pixels = centroids[clusters]
        return compressed_pixels.reshape(self.original_image.shape)  # returns the reshaped image

    def get_image(self, k):
        self.last_k = k  # updates the last displayed k value
        return self.kmeans_images[k]  # returns the kmeans image depending on the value of k


# list of the image names
image_filenames = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"]

# storing ImageCompressor objects in a list
image_compressor_list = []
for filename in image_filenames:
    image = Image.open(filename)
    image_array = np.array(image) / 255.0
    image_compressor_list.append(ImageCompressor(image_array))

# Create figure and UI
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3)
ax.set_title("Image Compression with K-means")
ax.axis('off')

# Track current state
current_image_index = 0
current_k = 0

# Display initial image
img_display = ax.imshow(image_compressor_list[current_image_index].get_image(current_k))

# Cluster slider
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Clusters', 0, 10, valinit=current_k, valstep=1)


def update_slider(val):
    global current_k
    current_k = int(val)  # holds the current value of k
    # this sets the data for the image in the list based on the current index and changes it based on the current k
    img_display.set_data(image_compressor_list[current_image_index].get_image(current_k))
    fig.canvas.draw_idle()


def change_image(direction):
    global current_image_index, current_k
    if direction == 'next':
        current_image_index = (current_image_index + 1) % len(image_filenames)
    else:
        current_image_index = (current_image_index - 1) % len(image_filenames)

    # getting the last k value for the image
    current_k = image_compressor_list[current_image_index].last_k
    slider.set_val(current_k)  # updating the number that the slider is on

    # updating the image with the current k value
    img_display.set_data(image_compressor_list[current_image_index].get_image(current_k))
    fig.canvas.draw_idle()


# Buttons
ax_prev = plt.axes([0.1, 0.01, 0.1, 0.075])
ax_next = plt.axes([0.8, 0.01, 0.1, 0.075])
button_prev = Button(ax_prev, 'Previous')
button_next = Button(ax_next, 'Next')

button_prev.on_clicked(lambda event: change_image('prev'))
button_next.on_clicked(lambda event: change_image('next'))
slider.on_changed(update_slider)

plt.show()
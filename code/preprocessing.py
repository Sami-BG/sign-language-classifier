import tensorflow as tf
import pandas as pd


def read_images(file):
	"""
	Reads CSV File of SQUARE image.
	:param file: String path of file.
	:return: tensor of Labels, Images
	"""
	csv = pd.read_csv(file)
	labels = tf.convert_to_tensor(csv.iloc[:, 0])
	total_pixels = csv.shape[1] - 1
	# Detect size of square image
	edge = round(total_pixels ** 0.5)
	images_df = csv.iloc[:, 1:]
	images_scaled = images_df * (1.0 / 255)
	images = tf.reshape(tf.convert_to_tensor(images_scaled), [-1, edge, edge, 1])
	return images, labels


def patch_images(images, patch_size, channels=1):
	"""
	Takes an image and turns it into patches.
	:param images: Batch of images.
	:param channels: Channels of image (3 for RGB. In our case: 1)
	:param patch_size: Size n of each n x n patch. for 28x28 image, 7 or 4 is valid patch_size, something that does not
	divide cleanly is not.
	:return: tuple of patches (shape [num_images, num_patches, pixels_in_patch])
	"""
	patch_dim = channels * (patch_size ** 2)
	num_images = images.shape[0]
	patches = tf.image.extract_patches(images=images,
									   sizes=[1, patch_size, patch_size, 1],
									   strides=[1, patch_size, patch_size, 1],
									   rates=[1, 1, 1, 1],
									   padding='VALID')
	patches = tf.reshape(patches, [num_images, -1, patch_dim])
	return patches


def preprocess(path, patch_size, channels=1):
	images, labels = read_images(path)
	patches = patch_images(images, patch_size, channels)
	return patches, labels


if __name__ == '__main__':
	preprocess('../data/sign_mnist_train/sign_mnist_train.csv', patch_size=7)

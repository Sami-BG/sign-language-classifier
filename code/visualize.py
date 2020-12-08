import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import preprocess
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def visualize_image():
	raw_inputs = pd.read_csv('../data/sign_mnist_train.csv')
	input1 = raw_inputs.iloc[4][1:].array
	label1: int = raw_inputs.iloc[4][0]

	model_inputs, model_labels = preprocess('../data/sign_mnist_train/sign_mnist_train.csv', patch_size=7)
	model = tf.keras.models.load_model('./SavedModels')
	label_test = tf.argmax(tf.squeeze(model(tf.expand_dims(model_inputs[4], axis=0))))
	plt.title(f'Predicted:{str(label_test)} - True: {label1}')
	image_format = np.array(input1).reshape((28, 28))
	plt.imshow(image_format)
	plt.show()


def make_confusion_matrix():
	model_inputs, true_labels = preprocess('../data/sign_mnist_test.csv', patch_size=7)
	model = tf.keras.models.load_model('./SavedModels')
	predictions = np.array(tf.argmax(model(model_inputs), axis=1))
	true_labels = np.array(true_labels)

	cf_matrix = confusion_matrix(true_labels, predictions)
	normalized = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
	print(np.shape(cf_matrix))

	labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
			  "W", "X", "Y"]
	df = pd.DataFrame(normalized, columns=labels, index=labels)

	heat_map = sns.heatmap(df, vmin=0, vmax=1, linewidths=0.2)
	heat_map.set(title="Confusion Matrix", xlabel="Predicted Letter", ylabel="Actual Letter")
	plt.yticks(rotation=0)
	plt.show()
	plt.savefig()


if __name__ == '_main_':
	# visualize_image()
	make_confusion_matrix()
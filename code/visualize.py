import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import preprocess
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
		  "W", "X", "Y"]


def plot_loss(path, title):
	episodes = pd.read_csv(path)
	plt.plot(episodes['loss'].values)
	plt.plot(episodes['val_loss'].values)
	plt.title(title)
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'validation'])
	plt.show()


def plot_accuracy(path, title):
	episodes = pd.read_csv(path)
	plt.plot(episodes['accuracy'].values)
	plt.plot(episodes['val_accuracy'].values)
	plt.title(title)
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'validation'])
	plt.show()


def visualize_image(letter: int):
	raw_inputs = pd.read_csv('../data/sign_mnist_train.csv')

	input1_index = raw_inputs.index[raw_inputs['label'] == letter][0]
	input1 = raw_inputs.iloc[input1_index][1:].array
	label1: int = letter

	model_inputs, model_labels = preprocess('../data/sign_mnist_train/sign_mnist_train.csv', patch_size=7)
	model = tf.keras.models.load_model('./ViT_Baseline')
	label_test = tf.argmax(tf.squeeze(model(tf.expand_dims(model_inputs[input1_index], axis=0))))
	plt.title(f'Predicted:{labels[int(label_test)]} - True: {labels[label1]}')
	image_format = np.array(input1).reshape((28, 28))
	plt.imshow(image_format)
	plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
	cf_matrix = confusion_matrix(y_true, y_pred)
	normalized = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
	print(np.shape(cf_matrix))
	df = pd.DataFrame(normalized, columns=labels, index=labels)
	heat_map = sns.heatmap(df, vmin=0, vmax=1, linewidths=0.2)
	heat_map.set(title=title, xlabel="Predicted Letter", ylabel="Actual Letter")
	plt.yticks(rotation=0)
	plt.show()


if __name__ == '__main__':
	# Visualizing vision transformer results
	visualize_image(letter=0)
	model_inputs, true_labels = preprocess('../data/sign_mnist_test.csv', patch_size=7)
	model: tf.keras.models.Model = tf.keras.models.load_model('./ViT_Baseline')
	model.evaluate(model_inputs, true_labels)
	predictions = np.array(tf.argmax(model(model_inputs), axis=1))
	true_labels = np.array(true_labels)
	print(classification_report(true_labels, y_pred=predictions, target_names=labels))
	plot_confusion_matrix(y_true=true_labels, y_pred=predictions, title='Confusion Matrix for Vision Transformer')
	plot_loss('./vit_history.csv', 'Loss for Vision Transformer')
	plot_accuracy('./vit_history.csv', 'Accuracy for Vision Transformer')

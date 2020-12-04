import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import preprocess
import pandas as pd
import numpy as np

if __name__ == '__main__':
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




import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt
from visualize import make_confusion_matrix as plot_confusion_matrix

TRAIN_PATH = '../data/sign_mnist_train.csv'
TEST_PATH = '../data/sign_mnist_test.csv'
EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


def read_data(path):
	data = pd.read_csv(path)
	labels = data.iloc[:, 0]
	inputs = data.iloc[:, 1:]
	inputs = tf.reshape(inputs, shape=(len(inputs), 28, 28, 1))
	inputs /= 255
	return inputs, labels


def setup_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(25, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				  optimizer=tf.keras.optimizers.Adam(.005), metrics=['accuracy'])
	return model


def train_model(base_model, x, y):
	episodes = base_model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
	base_model.save('./CNN')
	# Save history as CSV to plot later
	hist_df = pd.DataFrame(episodes.history)
	# Save to csv:
	history_csv_path = 'cnn_history.csv'
	with open(history_csv_path, mode='w') as f:
		hist_df.to_csv(f)

	return episodes


def plot_loss(path):
	episodes = pd.read_csv(path)
	plt.plot(episodes['loss'].values)
	plt.plot(episodes['val_loss'].values)
	plt.title("Loss for CNN Implementation")
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'validation'])
	plt.show()


def plot_accuracy(path):
	episodes = pd.read_csv(path)
	plt.plot(episodes['accuracy'].values)
	plt.plot(episodes['val_accuracy'].values)
	plt.title("Accuracy for CNN Implementation")
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'validation'])
	plt.show()


def main():
	x_train, y_raw = read_data(TRAIN_PATH)
	y_train = to_categorical(y_raw)
	x_test, y_test_raw = read_data(TEST_PATH)
	y_test = to_categorical(y_test_raw)
	model = setup_model()
	model = tf.keras.models.load_model('./CNN')
	# episodes = train_model(model, x_train, y_train)
	plot_loss(path='./cnn_history.csv')
	plot_accuracy(path='./cnn_history.csv')
	predictions = model.predict_classes(x_test)
	model.evaluate(x_test, y_test)
	plot_confusion_matrix(y_true=y_test_raw.values, y_pred=predictions)


if __name__ == '__main__':
	main()

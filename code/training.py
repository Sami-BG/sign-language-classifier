from vision_transformer import VisionTransformer
import tensorflow as tf
from preprocessing import preprocess

if __name__ == '__main__':
	# None of this is tuned at all. Just arbitrary numbers

	inputs, labels = preprocess('../data/sign_mnist_train/sign_mnist_train.csv', patch_size=7)
	model: tf.keras.Model = VisionTransformer(mlp_dim=128, num_classes=25,
											  model_dim=64, num_heads=4, num_patches=16, dropout=True,
							  dropout_amount=0.1, encoding_layers=4)
	model.compile(optimizer=tf.optimizers.Adam(),
				  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	model.fit(inputs, labels, batch_size=16, validation_split=0.2, epochs=15)
	model.save('./SavedModels/')

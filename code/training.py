from vision_transformer import VisionTransformer
import tensorflow as tf
from preprocessing import preprocess
from tensorflow.keras.models import Model
import tensorflow_model_optimization as tfmot


prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


def train_model(inputs, labels, name):
	model: tf.keras.Model = VisionTransformer(mlp_dim=64, num_classes=25,
											  model_dim=32, num_heads=4, num_patches=16, dropout=True,
											  dropout_amount=0.1, encoding_layers=2)
	model.compile(optimizer=tf.optimizers.Adam(),
				  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	model.fit(inputs, labels, batch_size=16, validation_split=0.2, epochs=25)
	model.evaluate(x=test_inputs, y=test_labels)
	model.save(f'./{name}/', overwrite=True)
	model.save_weights(f'./{name}_weights/', overwrite=True)
	return model


def prune_model(base_model, name):
	"""
	Currently does not work, as VisionTransformer class cannot be cloned. It cannot be cloned because it is not
	serializable. It is not serializable because tf.Variables are not serializable.
	:param base_model:
	:param name:
	:return:
	"""
	path = f'./{name}'

	# Helper function uses `prune_low_magnitude` to make only the
	# Dense layers train with pruning.
	def apply_pruning_to_dense(layer):
		if isinstance(layer, tf.keras.layers.Dense):
			return tfmot.sparsity.keras.prune_low_magnitude(layer)
		return layer

	# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
	# to the layers of the model.
	model_for_pruning = tf.keras.models.clone_model(
		base_model,
		clone_function=apply_pruning_to_dense,
	)

	# Include optimizer because pruning removes it.
	model_for_pruning.save(path, include_optimizer=True)

	# Need to recompile because optimizer removed.
	model_for_pruning.compile(
		loss=tf.keras.losses.categorical_crossentropy,
		optimizer='adam',
		metrics=['accuracy']
	)

	return model_for_pruning


def train_with_pruning(pruned_model: Model, inputs, labels, name):

	path = f'./{name}_Pruned_WithTraining'

	callbacks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		# Log sparsity and other metrics in Tensorboard.
		tfmot.sparsity.keras.PruningSummaries(log_dir=path + '_Summaries/')
	]

	pruned_model.compile(
		loss=tf.keras.losses.categorical_crossentropy,
		optimizer='adam',
		metrics=['accuracy']
	)

	pruned_model.fit(
		inputs,
		labels,
		callbacks=callbacks,
		epochs=10,
	)

	pruned_model.save(path, include_optimizer=True)

	return pruned_model


if __name__ == '__main__':
	# None of this is tuned at all. Just arbitrary numbers
	inputs, labels = preprocess('../data/sign_mnist_train/sign_mnist_train.csv', patch_size=7)
	test_inputs, test_labels = preprocess('../data/sign_mnist_test/sign_mnist_test.csv', patch_size=7)
	baseline = train_model(inputs, labels, name='ViT_Baseline')
	# baseline = tf.keras.models.load_model('./SavedModels/')
	baseline.evaluate(test_inputs, test_labels)

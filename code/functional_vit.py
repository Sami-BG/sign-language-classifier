import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Input
import tensorflow_model_optimization as tfmod
from transformer_block import Transformer


def setup_vit(batch_size,
			  mlp_dim,
			  num_classes,
			  model_dim,
			  num_heads,
			  num_patches,
			  patch_area=49,
			  dropout=True,
			  dropout_amount=0.1,
			  encoding_layers=5):
	args = [model_dim, num_heads, 2, dropout_amount]
	inputs = Input(batch_shape=(batch_size, num_patches, patch_area))
	projected = Dense(model_dim, name='patch_embed')(inputs)
	class_embedding = tf.Variable(name='class_embedding', shape=(1, 1, model_dim),
								  initial_value=tf.random.normal(shape=[1, 1, model_dim]))
	positional_embedding = tf.Variable(name='positional_embedding', shape=(1, num_patches + 1, model_dim),
									   initial_value=tf.random.normal(shape=[1, num_patches + 1, model_dim]))
	# Expand dimensions of class embedding to batch of patches
	class_embedding = tf.broadcast_to(class_embedding, [tf.shape(inputs)[0], 1, model_dim])
	outputs0 = tf.add(tf.concat([class_embedding, projected], axis=1), positional_embedding)
	# Create encoding blocks
	outputs1 = Transformer(args[0], args[1], args[2], args[3], name='encoder_1')(outputs0)
	outputs2 = Transformer(args[0], args[1], args[2], args[3], name='encoder_2')(outputs1)
	outputs3 = Transformer(args[0], args[1], args[2], args[3], name='encoder_3')(outputs2)
	outputs4 = Transformer(args[0], args[1], args[2], args[3], name='encoder_4')(outputs3)
	outputs5 = Transformer(args[0], args[1], args[2], args[3], name='encoder_5')(outputs4)

	normalized = LayerNormalization(epsilon=1e-6, name='vit_norm')(outputs5)
	dense1 = Dense(mlp_dim, activation=tf.keras.activations.relu, name='vit_dense1')(normalized)
	drop1 = Dropout(dropout_amount, name='vit_dropout')(dense1)
	dense2 = Dense(num_classes, name='vit_dense2')(drop1)

	model = tf.keras.Model(inputs, dense2, name='vit_functional')
	return model


if __name__ == '__main__':
	# TODO: TF.Variables are not serializable in functional models. RIP
	functional_model = setup_vit(batch_size=16, mlp_dim=128, num_classes=25,
								 model_dim=64, num_heads=4,
								 num_patches=16, dropout=True,
								 dropout_amount=0.1, encoding_layers=5)
	cloned = tf.keras.models.clone_model(functional_model)
	a = 2

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout
import tensorflow_model_optimization as tfmod
from transformer_block import Transformer


class VisionTransformer(tf.keras.models.Model):
	def __init__(self, mlp_dim, num_classes, model_dim, num_heads, num_patches,
				 dropout=True, dropout_amount=0.1, encoding_layers=5):
		super(VisionTransformer, self).__init__()
		# Storing all inputs for config
		self.mlp_dim = mlp_dim
		self.num_classes = num_classes
		self.model_dim = model_dim
		self.num_heads = num_heads
		self.num_patches = num_patches
		self.dropout = dropout
		self.dropout_amount = dropout_amount
		self.encoding_layers = encoding_layers
		# Define MLP for head
		self.MLP_norm = LayerNormalization(epsilon=1e-6, name='vit_norm')
		self.MLP_W1 = Dense(mlp_dim, activation=tf.keras.activations.relu, name='vit_dense1')
		self.MLP_Dropout = Dropout(dropout_amount, name='vit_dropout')
		# NOTE! DON'T USE SOFTMAX! Gradients vanish
		self.MLP_W2 = Dense(num_classes, name='vit_dense2')

		self.encoding_blocks = [Transformer(embedded_dim=model_dim, number_heads=num_heads,
											mlp_dims=2, dropout=dropout_amount, name=f'encoding_{i}')
								for i in range(0, encoding_layers)]
		# Embeds patches in higher dimension
		self.patch_embeddings = Dense(model_dim, name='patch_embed')
		# Positional embeddings take the image patches + class embedding
		self.positional_embedding = self.add_weight(name="positional_embedding", shape=(1, num_patches + 1, model_dim))
		# Class embedding to infer from
		self.class_embedding = self.add_weight(name="class_embedding", shape=(1, 1, model_dim))

	# Now it doesn't override get_config I suppose!
	def not_get_config(self):
		config = super(VisionTransformer, self).get_config()
		config.update({
			'mlp_dim': self.mlp_dim,
			'num_classes': self.num_classes,
			'model_dim': self.model_dim,
			'num_heads': self.num_heads,
			'num_patches': self.num_patches,
			'dropout': self.dropout,
			'dropout_amount': self.dropout_amount,
			'encoding_layers': self.encoding_layers
		})

	# Also commented this out cause it does not work
	@classmethod
	def not_from_config(cls, config, custom_objects=None):
		return cls(**config)

	@tf.function
	def call(self, inputs, training):
		projected = self.patch_embeddings(inputs)
		# Expand dimensions of class embedding to batch of patches
		class_embedding = tf.broadcast_to(self.class_embedding, [tf.shape(inputs)[0], 1, self.model_dim])
		outputs = tf.concat([class_embedding, projected], axis=1) + self.positional_embedding
		for layer in self.encoding_blocks:
			# Pass training for dropout layers during inference time
			outputs = layer(outputs, training)

		# Use the result of applying model to class embedding for final classification result
		outputs = self.MLP_norm(outputs[:, 0])
		outputs = self.MLP_W1(outputs)
		outputs = self.MLP_Dropout(outputs)
		outputs = self.MLP_W2(outputs)

		return outputs

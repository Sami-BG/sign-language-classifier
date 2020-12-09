import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout
import tensorflow_model_optimization as tfmod


class Transformer(tf.keras.layers.Layer, tfmod.sparsity.keras.PrunableLayer):
	def __init__(self, embedded_dim, number_heads, mlp_dims, dropout=0.1, name=None):
		if not name:
			name = 'TransformerBlock'
		super(Transformer, self).__init__(name=name)
		# Saved for config
		self.embedded_dim = embedded_dim
		self.number_heads = number_heads
		self.mlp_dims = mlp_dims
		self.dropout = dropout
		# TF Nightly Implementation. Not pruned.
		self.MultiHeadModel = tf.keras.layers.MultiHeadAttention(num_heads=number_heads,
																 key_dim=embedded_dim,
																 value_dim=embedded_dim,
																 name=f'{self.name}_encoder_mha')

		self.MLP_W1 = Dense(mlp_dims, activation=tf.keras.activations.relu, name='encoder_mlp_dense1')
		self.MLP_Dropout1 = Dropout(dropout, name='encoder_mlp_dropout1')
		self.MLP_W2 = Dense(embedded_dim, name='encoder_mlp_dense2')
		self.MLP_Dropout2 = Dropout(dropout, name='encoder_mlp_dropout2')

		# Omitted from pruning.
		self.MultiHeadValue = Dense(embedded_dim, name='encoder_MHV')

		self.drop1 = Dropout(dropout, name='encoder_drop1')
		self.drop2 = Dropout(dropout, name='encoder_drop2')
		self.norm = LayerNormalization(epsilon=1e-5, name='encoder_norm')

	# This does not work, so for the sake of time, it's being omitted :(
	def not_get_config(self):
		config = super(Transformer, self).get_config()
		config.update({
			'embedded_dim': self.embedded_dim,
			'number_heads': self.number_heads,
			'mlp_dims': self.mlp_dims,
			'dropout': self.dropout
		})

	# Note: This is the default implementation, so we technically don't need to have this here. But also, doesn't work.
	#  so not_from_config!
	@classmethod
	def not_from_config(cls, config):
		return cls(**config)

	def get_prunable_weights(self):
		return [self.MLP_W1.kernel, self.MLP_W2.kernel]

	@tf.function
	def call(self, inputs, training):
		norm1 = self.norm(inputs)
		attention = self.MultiHeadModel(norm1, norm1)
		dropped1 = self.drop1(attention, training)

		attention = dropped1 + inputs

		mlp = self.MLP_W1(attention)
		mlp = self.MLP_Dropout1(mlp)
		mlp = self.MLP_W2(mlp)
		mlp = self.MLP_Dropout2(mlp)[0]
		mlp_dropped = self.drop2(mlp, training)

		outputs = mlp_dropped + attention

		return outputs

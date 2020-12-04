import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout


class Transformer(tf.keras.layers.Layer):
	def __init__(self, embedded_dim, number_heads, mlp_dims, dropout=0.1):
		super(Transformer, self).__init__()
		self.MultiHeadModel = tf.keras.layers.MultiHeadAttention(num_heads=number_heads,
																 key_dim=embedded_dim,
																 value_dim=embedded_dim, name='encoder_mha')
		self.MLP_W1 = Dense(mlp_dims, activation=tf.keras.activations.relu, name='encoder_mlp_dense1')
		self.MLP_Dropout1 = Dropout(dropout, name='encoder_mlp_dropout1')
		self.MLP_W2 = Dense(embedded_dim, name='encoder_mlp_dense2')
		self.MLP_Dropout2 = Dropout(dropout, name='encoder_mlp_dropout2')


		self.MultiHeadValue = Dense(embedded_dim, name='encoder_MHV')

		self.drop1 = Dropout(dropout, name='encoder_drop1')
		self.drop2 = Dropout(dropout, name='encoder_drop2')
		self.norm = LayerNormalization(epsilon=1e-5, name='encoder_norm')

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


class VisionTransformer(tf.keras.models.Model):
	def __init__(self, mlp_dim, num_classes, model_dim, num_heads, num_patches,
				 dropout=True, dropout_amount=0.1, encoding_layers=5):
		super(VisionTransformer, self).__init__()
		self.model_dim = model_dim

		# Define MLP for head
		self.MLP_norm = LayerNormalization(epsilon=1e-6, name='vit_norm')
		self.MLP_W1 = Dense(mlp_dim, activation=tf.keras.activations.relu, name='vit_dense1')
		self.MLP_Dropout = Dropout(dropout_amount, name='vit_dropout')
		self.MLP_W2 = Dense(num_classes, name='vit_dense2')

		self.encoding_blocks = [Transformer(embedded_dim=model_dim, number_heads=num_heads,
											mlp_dims=2, dropout=dropout_amount)
								for _ in range(0, encoding_layers)]
		# Embeds patches in higher dimension
		self.patch_embeddings = Dense(model_dim, name='patch_embed')
		# Positional embeddings take the image patches + class embedding
		self.positional_embedding = self.add_weight(name="positional_embedding", shape=(1, num_patches + 1, model_dim))
		# Class embedding to infer from
		self.class_embedding = self.add_weight(name="class_embedding", shape=(1, 1, model_dim))

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

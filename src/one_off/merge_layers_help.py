	def run(self):
		# network definition
		input_shape = training_data.shape[2:]
		base_network = self.create_base_network(input_shape)

		input_a = Input(shape=input_shape, name='input_a')
		input_b = Input(shape=input_shape, name='input_b')

		processed_a = base_network(input_a)
		processed_b = base_network(input_b)

		distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

		auxiliary_input = Input(shape=(1,), name='auxiliary_input')
		combined_layer = keras.layers.concatenate([distance, auxiliary_input])
		main_output = Dense(1, activation='sigmoid', name='main_output')(combined_layer)

		# originally: 'distance' was a scalar value ranging from 0 to 1.7ish, and was
		#     being compared to the goal (0 or 1)
		# goal: instead, i'd like the distance to be based not only on the euc. distance
		#     of the 2 CNN's, but also having combined a separate input layer (let's say 3 values)
		#     with this, so that all 4 values feed into a new sigmoid and yield a value 0 to 1,
		#     and we'd still use the same loss function (self.constrastive_loss)

		model = Model([input_a, input_b, auxiliary_input], main_output)
		model.compile(loss=self.contrastive_loss, optimizer=opt)
		model.fit({'input_a': np.array(training_data[0][:, 0]),
					'input_b': np.array(training_data[0][:, 1]),
					'auxiliary_input': np.array(aux_data)},
				{'main_output': training_labels}, 
				batch_size=self.args.batchSize,
				epochs=self.args.numEpochs,
				validation_data=([dev_data[:, 0], dev_data[:, 1]], dev_labels))
	def eucl_dist_output_shape(self, shapes):
		shape1, shape2 = shapes
		return (shape1[0], 1)

	def contrastive_loss(self, y_true, y_pred):
		margin = 1
		return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

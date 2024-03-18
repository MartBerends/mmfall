from tensorflow.keras.layers import Input, Dense, Lambda, RepeatVector, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from transformers import TransformerEncoder, TransformerDecoder

def HVRAE_train_with_transformer(self, train_data, test_data):
    n_frames = 10
    n_points = 64
    n_features = 4
    n_intermediate = 64
    n_latentdim = 16

    inputs = Input(shape=(n_frames, n_points, n_features))
    flattened_input = Reshape((n_frames * n_points, n_features))(inputs)

    # Replace RNN with Transformer Encoder
    encoder_output = TransformerEncoder(num_layers=2, d_model=n_latentdim, num_heads=4, dff=n_intermediate, input_vocab_size=n_points * n_frames, maximum_position_encoding=n_points * n_frames)(flattened_input)

    # Sampling mechanism
    Z_mean = Dense(n_latentdim, activation=None, name='qzx_mean')(encoder_output)
    Z_log_var = Dense(n_latentdim, activation=None, name='qzx_log_var')(encoder_output)
    Z = Lambda(sampling)([Z_mean, Z_log_var])

    # Replace RNN with Transformer Decoder
    decoder_output = TransformerDecoder(num_layers=2, d_model=n_latentdim, num_heads=4, dff=n_intermediate, target_vocab_size=n_points * n_frames, maximum_position_encoding=n_points * n_frames,)(Z)

    # Output layers
    pXz_mean = Dense(n_features, activation=None)(decoder_output)
    pXz_logvar = Dense(n_features, activation=None)(decoder_output)

    pXz = Concatenate()([pXz_mean, pXz_logvar])
    pXz = Reshape((n_frames, n_points, n_features * 2))(pXz)

    model = Model(inputs, pXz)

    def HVRAE_loss(y_true, y_pred):
        # Define loss function as before
        pass

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss=HVRAE_loss)

    model.fit(train_data, train_data,
              epochs=5,
              batch_size=8,
              shuffle=False,
              validation_data=(test_data, test_data))

    model.save(self.model_dir + 'HVRAE_mdl_online_with_transformer.h5')
    print("INFO: Training is done!")

def sampling(args):
    # Define sampling function as before
    pass



class autoencoder_mdl:
    def __init__(self, model_dir):
        self.model_dir = model_dir


    def transmodel_train(self, train_data, test_data):
    #Input
        n_frames = 10
        n_points = 64
        n_features = 4
        n_latentdim = 512 #todo why am I using these numbers?
        n_heads = 4  
        inputs = Input(shape=(n_frames, n_points, n_features))
        outputs = Input(shape=(n_frames, n_points, n_features))

        #positional embedding
        def positional_encoding(max_seq_length, d_model):
            pos_enc = np.zeros((max_seq_length, d_model))
            position = np.arange(0, max_seq_length)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            pos_enc[:, 0::2] = np.sin(position * div_term)
            pos_enc[:, 1::2] = np.cos(position * div_term)
            return tf.convert_to_tensor(pos_enc, dtype=tf.float32)

        max_seq_length = 10
        d_model = 512
        self.pos_enc = positional_encoding(max_seq_length, d_model)
        input_flatten = TimeDistributed(Flatten())(inputs)
        def transformer_encoder(inputs):
            print("inside encoder")
            print("Shape of input flatten:", input_flatten.shape)
            
            attention_output = MultiHeadAttention(num_heads=n_heads, key_dim=n_latentdim)(input_flatten, input_flatten) #multihead
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output + input_flatten) #adding and normalization with residual connection
            ffn_output = TimeDistributed(Dense(256, activation='relu'))(attention_output) #feed forward network
            ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output) #add & norm again
            print("ffn_outputshape", ffn_output.shape)
            print("inputsshape", inputs.shape)
            return ffn_output
       
        encoder_output = transformer_encoder(inputs)
        Z_mean = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(encoder_output)
        Z_log_var = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(encoder_output)


        #building the model
        self.TRANS_mdl = Model(inputs, outputs)
        print(self.TRANS_mdl.summary())
        self.TRANS_mdl.compile(optimizer=adam, loss=mse)
        # Train the model


        self.TRANS_mdl.fit(train_data, train_data, # Train on the normal training set
                epochs=5,
                batch_size=80,
                shuffle=False,
                validation_data=(test_data, test_data), # Testing on the normal tesing dataset
                callbacks=[TensorBoard(log_dir=(self.model_dir + "/../model_history/TRANS_online"))])
        self.TRANS_mdl.save(self.model_dir + 'RAE_TRANS_online.h5')
       
        print("INFO: Training is done!")
        print("*********************************************************************")

model = autoencoder_mdl(model_dir = (project_path + 'saved_model/'))

model.transmodel_train(train_data, test_data)











    
class TransformerAnomalyDetection(tf.keras.Model):
    def __init__(self, num_heads, d_model, num_layers, d_ff, input_shape):
        super(TransformerAnomalyDetection, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.input_shape = input_shape

        # Positional encoding
        self.positional_encoding = self._get_positional_encoding()

        # Transformer layers
        self.transformer_layers = [
            self._get_transformer_layer() for _ in range(num_layers)
        ]

        # Output layer
        self.output_layer = Dense(1, activation='sigmoid')

    def _get_positional_encoding(self):
        # Create positional encoding matrix
        pos_enc = tf.range(self.input_shape[1], dtype=tf.float32)
        pos_enc = pos_enc[:, tf.newaxis] / tf.pow(10000.0, 2 * tf.range(self.d_model, dtype=tf.float32) / self.d_model)
        pos_enc[:, 0::2] = tf.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = tf.cos(pos_enc[:, 1::2])
        return pos_enc

    def positional_encoding(max_seq_length, d_model):
        pos_enc = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return tf.convert_to_tensor(pos_enc, dtype=tf.float32)


    def _get_transformer_layer(self):
        return tf.keras.Sequential([
            MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model),
            LayerNormalization(epsilon=1e-6),
            Dense(self.d_ff, activation='relu'),
            Dense(self.d_model),
            LayerNormalization(epsilon=1e-6)
        ])

    def call(self, inputs):
        # Add positional encoding
        inputs += self.positional_encoding[:, :self.input_shape[1], :]

        # Pass through transformer layers
        for layer in self.transformer_layers:
            inputs = layer(inputs)

        # Aggregate across time steps
        aggregated = tf.reduce_mean(inputs, axis=1)

        # Output layer
        output = self.output_layer(aggregated)
        return output

# Example usage
input_shape = (None, 4)  # 4-dimensional point cloud data
model = TransformerAnomalyDetection(num_heads=4, d_model=64, num_layers=2, d_ff=128, input_shape=input_shape)
model.build(input_shape)
model.summary()












def create_transformer_model(input_shape, num_heads, d_model, num_layers, dropout_rate):
    """
    Creates a transformer model with self-attention for anomaly detection.

    Args:
        input_shape (tuple): Shape of the input data (60816, 10, 64, 4).
        num_heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        num_layers (int): Number of transformer layers.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model (tf.keras.Model): Transformer model.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    # Positional embedding
    def positional_encoding(max_seq_length, d_model):
        pos_enc = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return tf.convert_to_tensor(pos_enc, dtype=tf.float32)
    # Flatten the input data
    # Add positional embedding
    x += positional_encoding(max_seq_length, d_model)


    for _ in range(num_layers):
        # Multi-head self-attention
        x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)([x, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(rate=dropout_rate)(x)

        # Feed-forward neural network
        x_ffn = Dense(units=d_model, activation="relu")(x)
        x_ffn = Dense(units=d_model)(x_ffn)
        x = LayerNormalization(epsilon=1e-6)(x + x_ffn)
        x = Dropout(rate=dropout_rate)(x)

    output = Dense(units=1, activation="sigmoid")(x)

    # Create the model
    model = Model(inputs=inputs, outputs=output)
    return model

max_seq_length = 10
input_shape = (10, 64, 4)
num_heads = 8
d_model = 64
num_layers = 4
dropout_rate = 0.1
#Create the transformer model
transformer_model = create_transformer_model(input_shape, num_heads, d_model, num_layers, dropout_rate)

# Compile the model
transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Print model summary
transformer_model.summary()
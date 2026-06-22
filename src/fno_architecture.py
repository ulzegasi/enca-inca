import tensorflow as tf


class SpectralConv1D(tf.keras.layers.Layer):
    """1D spectral convolution used by Fourier Neural Operator blocks."""

    def __init__(self, out_channels, modes, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.modes = modes

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        scale = 1.0 / max(1, in_channels * self.out_channels)
        self.weight_real = self.add_weight(
            name="weight_real",
            shape=(self.modes, in_channels, self.out_channels),
            initializer=tf.keras.initializers.RandomNormal(stddev=scale),
            trainable=True,
        )
        self.weight_imag = self.add_weight(
            name="weight_imag",
            shape=(self.modes, in_channels, self.out_channels),
            initializer=tf.keras.initializers.RandomNormal(stddev=scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        n_time = tf.shape(x)[1]
        x_ft = tf.signal.rfft(tf.transpose(x, perm=[0, 2, 1]))
        x_ft = tf.transpose(x_ft, perm=[0, 2, 1])

        weights = tf.complex(self.weight_real, self.weight_imag)
        x_ft_low = x_ft[:, : self.modes, :]
        out_ft_low = tf.einsum("bmi,mio->bmo", x_ft_low, weights)

        n_freq = tf.shape(x_ft)[1]
        pad_modes = n_freq - self.modes
        out_ft = tf.pad(out_ft_low, [[0, 0], [0, pad_modes], [0, 0]])
        out_ft = tf.transpose(out_ft, perm=[0, 2, 1])
        x_out = tf.signal.irfft(out_ft, fft_length=[n_time])
        return tf.transpose(x_out, perm=[0, 2, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels, "modes": self.modes})
        return config


class FNOBlock1D(tf.keras.layers.Layer):
    """Spectral convolution plus pointwise mixing."""

    def __init__(self, width, modes, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.modes = modes
        self.activation_name = activation
        self.spectral = SpectralConv1D(width, modes)
        self.pointwise = tf.keras.layers.Conv1D(width, kernel_size=1)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, x):
        x = self.spectral(x) + self.pointwise(x)
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "modes": self.modes,
                "activation": self.activation_name,
            }
        )
        return config


def _time_coordinate_layer(len_timeseries, name):
    time_grid = tf.linspace(0.0, 1.0, len_timeseries)
    time_grid = tf.reshape(time_grid, [1, len_timeseries, 1])
    return tf.keras.layers.Lambda(
        function=lambda x: tf.tile(tf.cast(time_grid, x.dtype), [tf.shape(x)[0], 1, 1]),
        name=name,
    )


def build_fno_encoder_decoder(
    *,
    len_timeseries,
    ndims_latent,
    num_noise_channels,
    fno_width=64,
    fno_modes=32,
    fno_layers=4,
    use_time_coordinate=True,
):
    """Rebuild the FNO ENCA encoder/decoder used by train_FNO_model3.py."""
    num_input_channels = 1
    fno_modes = min(int(fno_modes), int(len_timeseries) // 2 + 1)
    if fno_modes < 1:
        raise ValueError("fno_modes must be at least 1.")

    x_input = tf.keras.layers.Input(
        shape=[len_timeseries, num_input_channels], name="x_observation"
    )
    x = x_input
    if use_time_coordinate:
        t = _time_coordinate_layer(len_timeseries, name="encoder_time_coordinate")(x_input)
        x = tf.keras.layers.Concatenate(axis=-1, name="encoder_concat_time")([x, t])
    x = tf.keras.layers.Dense(fno_width, activation=None, name="encoder_lift")(x)
    for i in range(fno_layers):
        x = FNOBlock1D(fno_width, fno_modes, name=f"encoder_fno_block_{i+1}")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(fno_width, activation="gelu", name="encoder_dense")(x)
    z = tf.keras.layers.Dense(ndims_latent, activation=None, name="latent_space")(x)
    encoder = tf.keras.Model(inputs=x_input, outputs=z)

    latent_mappings = tf.keras.layers.Input(
        shape=[ndims_latent], name="latent_representations"
    )
    noise_vectors = tf.keras.layers.Input(
        shape=[len_timeseries, num_noise_channels], name="noise_vectors"
    )
    tile_layer = tf.keras.layers.Lambda(
        function=lambda a: tf.tile(tf.expand_dims(a, axis=1), multiples=[1, len_timeseries, 1]),
        name="tile_latent_space",
    )
    inputs = [tile_layer(latent_mappings), noise_vectors]
    if use_time_coordinate:
        t = _time_coordinate_layer(len_timeseries, name="decoder_time_coordinate")(noise_vectors)
        inputs.append(t)
    y = tf.keras.layers.Concatenate(axis=-1, name="concatenate_noise_latent_and_time")(inputs)
    y = tf.keras.layers.Dense(fno_width, activation=None, name="decoder_lift")(y)
    for i in range(fno_layers):
        y = FNOBlock1D(fno_width, fno_modes, name=f"decoder_fno_block_{i+1}")(y)
    y = tf.keras.layers.Dense(fno_width, activation="gelu", name="decoder_dense")(y)
    y = tf.keras.layers.Dense(units=num_input_channels, activation=None, name="pred")(y)
    y = tf.keras.layers.Reshape([len_timeseries, num_input_channels], name="output_shape")(y)
    decoder = tf.keras.Model(inputs=(latent_mappings, noise_vectors), outputs=y)

    return encoder, decoder

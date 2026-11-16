
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Adding this separately because It makes the attention layer reusable and ensures that the saved .keras model can be loaded reliably during later evaluation.

@keras.utils.register_keras_serializable(
    package="CSI_HAR"
)
class AdditiveTemporalAttention(layers.Layer):
    """
    Additive attention over temporal features.

    Input:
        (batch, timesteps, features)

    Output:
        (batch, features)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(
            1,
            activation="tanh",
            name="attention_score"
        )

    def call(self, inputs):
        # Calculate attention scores for each time step.
        scores = self.score_layer(inputs)

        # Normalize scores across the temporal dimension.
        attention_weights = tf.nn.softmax(
            scores,
            axis=1
        )

        # Weighted sum of temporal features.
        context_vector = tf.reduce_sum(
            inputs * attention_weights,
            axis=1
        )

        return context_vector

    def get_config(self):
        return super().get_config()

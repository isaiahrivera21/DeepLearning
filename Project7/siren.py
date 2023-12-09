import tensorflow as tf
from linear import Linear

## Citation: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=wt0akc6oiJgu

class Siren(tf.Module):
    def __init__(self, num_inputs, hidden_layer_width, num_outputs, num_hidden_layers):
        ## Initlization for first layer:
        self.firstLayer = Linear(
            num_inputs, hidden_layer_width, first_layer=True, bias=True
        )

        ## initilization for the other layers:
        self.hiddenLayers = [
            Linear(hidden_layer_width, hidden_layer_width, first_layer=False)
            for _ in range(num_hidden_layers)
        ]

        ## Final Layer
        self.finalLayer = Linear(
            hidden_layer_width, num_outputs, first_layer=False, bias=True
        )

    def __call__(self, X):
        X = tf.constant(X)
        X = tf.Variable(X, trainable=True)

        layer = tf.math.sin(30 * self.firstLayer(X))
        for hidden_layer in self.hiddenLayers:
            layer = tf.math.sin(30 * hidden_layer(layer))
        return (self.finalLayer(layer)), X

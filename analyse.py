import pickle
activations = pickle.load("checkpoints/mnist_cnn_vanilla.activations.11.6.pickle")
activations = pickle.load(open("checkpoints/mnist_cnn_vanilla.activations.11.6.pickle", 'rb'))
[x.shape for x  in activations
        ]
np.his(activations[0])
import numpy as np
np.histogram(activations[0])


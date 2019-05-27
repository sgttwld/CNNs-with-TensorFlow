# CNNs-with-tensorflow
Different levels of abstraction in convolutional neural network implementations with TensorFlow.

* [Convolutional layer from scratch](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/1_CNN_fromscratch.py): Convolutional neural network implementation with convolutional and pooling layers built from scratch with core tensorflow.

* [Low-level tensorflow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/2_CNN_lowlevel.py): Convolutional neural network using `tf.nn.conv2d` and `tf.nn.avg_pool` with explicit definitions of weights, biases, and placeholders.

* [Mid-level tensorflow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/3_CNN_midlevel.py): Convolutional neural network using `tf.keras.layers` managing weights and biases for us, whereas placeholders and the session are still explicit.

* [High-level tensorflow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/4_CNN_highlevel.py): Convolutional neural network using `tf.keras.model.Sequential` (everything is managed).

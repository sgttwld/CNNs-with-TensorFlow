# CNNs-with-TensorFlow
__Note:__ The implementations are using tensorflow 1, see [tensorflow-2-simple-examples](https://github.com/sgttwld/tensorflow-2-simple-examples) for tensorflow 2 examples.

 Implementations of an exemplary convolutional neural network with TensorFlow 1 using APIs at different levels of abstraction.

* [Convolutional layer from scratch](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/1_CNN_fromscratch.py): Convolutional neural network implementation with convolutional and pooling layers built from scratch with core TensorFlow.

* [Low-level TensorFlow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/2_CNN_lowlevel.py): Convolutional neural network using `tf.nn.conv2d` and `tf.nn.avg_pool` with explicit definitions of weights, biases, and placeholders.

* [Mid-level TensorFlow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/3_CNN_midlevel.py): Convolutional neural network using `tf.keras.layers` managing weights and biases for us, whereas placeholders and the session are still explicit.

* [High-level TensorFlow](https://github.com/sgttwld/CNNs-with-tensorflow/blob/master/4_CNN_highlevel.py): Convolutional neural network using `tf.keras.model.Sequential` (everything is managed).

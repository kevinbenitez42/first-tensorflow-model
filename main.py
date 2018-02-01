from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
import process_cifar
import model
from model import cnn_model_fn

from IPython.display import display

def main():
    cifar_list = process_cifar.download_data()

    training_set   = process_cifar.process_training_set(cifar_list)
    test_set       = process_cifar.process_test_set(cifar_list)

    train_data, train_labels = process_cifar.training_set_to_nparray(training_set)
    eval_data , eval_labels  = process_cifar.test_set_to_nparry(test_set)

    train_data  = train_data.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    eval_data   = eval_data.astype(np.float32)
    eval_labels = eval_labels.astype(np.float32)

    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model"
    )

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log, every_n_iter=50
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

    mnist_classifier.train(
    input_fn=train_input_fn,
    steps=2000,
    hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    main()

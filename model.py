import numpy as np
import tensorflow as tf
import pandas as pd


def convolution_layer(input_layer,filters,kernel_size,padding, activation):
    return tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation
        )


def pooling_layer(inputs,pool_size,strides):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=strides)


def cnn_model_fn(features,labels,mode):
    input_layer = tf.reshape(features['x'], [-1,32,32,1])

    conv1 = convolution_layer(input_layer,32,[5,5],
                             "same",tf.nn.relu)

    pool2 = pooling_layer(conv1,[2,2],2)

    conv2 = convolution_layer(pool2,64,[5,5],
                              "same",tf.nn.relu)

    pool2_flat = tf.reshape(conv2,[-1, 4*8 * 8 * 64 * 3])

    dense = tf.layers.dense(inputs=pool2_flat, units= 3072, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      "classes"      : tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=10)

    loss = tf.losses.softmax_cross_entropy(
         onehot_labels=onehot_labels,logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

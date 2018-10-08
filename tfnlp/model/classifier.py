import tensorflow as tf
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file

import tfnlp.common.constants as constants
from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.eval import log_trainable_variables
from tfnlp.layers.layers import input_layer
from tfnlp.layers.reduce import ConvNet


def classifier_model_func(features, mode, params):
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    input_dim = tf.shape(inputs)[-1]

    convs = [ConvNet.max_over_time_pooling_cnn(inputs, input_dim, params.config.state_size, width)
             for width in range(3, 6)]
    final_state = tf.concat(convs, axis=1)
    final_state = tf.layers.dropout(final_state, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = final_state

    target = params.extractor.targets[constants.LABEL_KEY]
    num_labels = target.vocab_size()

    with tf.variable_scope("inference_layer"):
        logits = tf.layers.dense(logits, num_labels, kernel_initializer=tf.zeros_initializer, name="softmax_projection")

    targets = None
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()
        targets = tf.identity(features[constants.LABEL_KEY], name=constants.LABEL_KEY)  # batch-length 1D tensor
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {constants.ACCURACY_METRIC_KEY: tf.metrics.accuracy(labels=targets, predictions=predictions)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        index_to_label = index_to_string_table_from_file(vocabulary_file=params.label_vocab_path,
                                                         default_value=target.unknown_word)
        predictions = index_to_label.lookup(tf.cast(predictions, dtype=tf.int64))
        export_outputs = {constants.PREDICT_KEY: PredictOutput(predictions)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs)

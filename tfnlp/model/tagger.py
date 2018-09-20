import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file

import tfnlp.common.constants as constants
from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.eval import SequenceEvalHook, SrlEvalHook, log_trainable_variables
from tfnlp.common.metrics import tagger_metrics
from tfnlp.layers.layers import encoder, input_layer


def tagger_model_func(features, mode, params):
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    outputs, output_size = encoder(features, inputs, mode, params.config)

    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, output_size], name="flatten_rnn_outputs_for_linear_projection")

    target = params.extractor.targets[constants.LABEL_KEY]
    num_labels = target.vocab_size()

    if params.config.zero_init:
        initializer = tf.zeros_initializer
    else:
        initializer = tf.random_normal_initializer(stddev=0.01)

    logits = tf.layers.dense(rnn_outputs, num_labels, kernel_initializer=initializer)
    logits = tf.reshape(logits, [-1, time_steps, num_labels], name="unflatten_logits")

    if params.config.crf:
        transition_matrix = tf.get_variable("transitions", [num_labels, num_labels])
    else:
        transition_matrix = tf.get_variable("transitions", [num_labels, num_labels],
                                            trainable=False, initializer=_create_transition_matrix(target))

    targets = None
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None
    evaluation_hooks = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()
        targets = tf.identity(features[constants.LABEL_KEY], name=constants.LABEL_KEY)

        if params.config.crf:
            log_likelihood, _ = crf_log_likelihood(logits, targets,
                                                   sequence_lengths=tf.cast(features[constants.LENGTH_KEY], tf.int32),
                                                   transition_params=transition_matrix)
            losses = -log_likelihood
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            mask = tf.sequence_mask(features[constants.LENGTH_KEY], name="padding_mask")
            losses = tf.boolean_mask(losses, mask, name="mask_padding_from_loss")
        loss = tf.reduce_mean(losses)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        predictions, _ = crf.crf_decode(logits, transition_matrix, tf.cast(features[constants.LENGTH_KEY], tf.int32))

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = tagger_metrics(predictions=tf.cast(predictions, dtype=tf.int64), labels=targets)
        eval_metric_ops[constants.ACCURACY_METRIC_KEY] = tf.metrics.accuracy(labels=targets, predictions=predictions)

        if params.config.type == constants.SRL_KEY:
            eval_hook = SrlEvalHook(
                tensors={
                    constants.LABEL_KEY: targets,
                    constants.PREDICT_KEY: predictions,
                    constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                    constants.MARKER_KEY: features[constants.MARKER_KEY],
                    constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX]
                },
                vocab=target)
            evaluation_hooks = [eval_hook]
        else:
            evaluation_hooks = [SequenceEvalHook(script_path=params.script_path,
                                                 gold_tensor=targets,
                                                 predict_tensor=predictions,
                                                 length_tensor=features[constants.LENGTH_KEY],
                                                 vocab=target)]

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
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)


def _create_transition_matrix(labels):
    """
    Return a numpy matrix to enforce valid transitions for IOB-style tagging problems.
    :param labels: label feature extractor
    """
    labels = [labels.index_to_feat(i) for i in range(len(labels.indices))]
    num_tags = len(labels)
    transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
    for i, prev_label in enumerate(labels):
        for j, curr_label in enumerate(labels):
            if i != j and curr_label[:2] == 'I-' and not prev_label == 'B' + curr_label[1:]:
                transition_params[i, j] = np.NINF
    return tf.initializers.constant(transition_params)

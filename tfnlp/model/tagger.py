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
    outputs, output_size, _ = encoder(features, inputs, mode, params.config)

    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, output_size], name="flatten_rnn_outputs_for_linear_projection")

    all_targets = params.extractor.targets
    transitions = {}
    logits = {}

    for target_key, target in all_targets.items():
        with tf.variable_scope("inference_layer_%s" % target_key):
            num_labels = target.vocab_size()
            if params.config.zero_init:
                initializer = tf.zeros_initializer
            else:
                initializer = tf.random_normal_initializer(stddev=0.01)

            logits[target_key] = tf.layers.dense(rnn_outputs, num_labels, kernel_initializer=initializer,
                                                 name="softmax_projection")
            logits[target_key] = tf.reshape(logits[target_key], [-1, time_steps, num_labels], name="unflatten_logits")

            if params.config.crf:
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels])
            else:
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels],
                                                          trainable=False, initializer=_create_transition_matrix(target))

    targets = {}
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None
    evaluation_hooks = None
    f1_score = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()

        def _loss(_target_key):
            targets[_target_key] = tf.identity(features[_target_key], name=_target_key)

            if params.config.crf:
                log_likelihood, _ = crf_log_likelihood(logits[_target_key], targets[_target_key],
                                                       sequence_lengths=tf.cast(features[constants.LENGTH_KEY], tf.int32),
                                                       transition_params=transitions[_target_key])
                _losses = -log_likelihood
            else:
                _losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[_target_key], labels=targets[_target_key])
                mask = tf.sequence_mask(features[constants.LENGTH_KEY], name="padding_mask_%s" % target_key)
                _losses = tf.boolean_mask(_losses, mask, name="mask_padding_from_loss_%s" % target_key)
            return tf.reduce_mean(_losses)

        if len(all_targets) == 1:
            loss = _loss(next(iter(all_targets.keys())))
        else:
            losses = [_loss(target_key) for target_key in all_targets.keys()]
            loss = tf.reduce_mean(tf.stack(losses))

        if params.config.type == constants.SRL_KEY:
            f1_score = tf.Variable(0, name=constants.SRL_METRIC_KEY, dtype=tf.float32, trainable=False)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        predictions, _ = crf.crf_decode(logits[constants.LABEL_KEY], transitions[constants.LABEL_KEY],
                                        tf.cast(features[constants.LENGTH_KEY], tf.int32))

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = tagger_metrics(predictions=tf.cast(predictions, dtype=tf.int64), labels=targets[constants.LABEL_KEY])
        eval_metric_ops[constants.ACCURACY_METRIC_KEY] = tf.metrics.accuracy(labels=targets[constants.LABEL_KEY],
                                                                             predictions=predictions)

        if params.config.type == constants.SRL_KEY:
            eval_metric_ops[constants.SRL_METRIC_KEY] = (tf.identity(f1_score), tf.identity(f1_score))
            eval_placeholder = tf.placeholder(dtype=tf.float32)
            eval_hook = SrlEvalHook(
                tensors={
                    constants.LABEL_KEY: targets[constants.LABEL_KEY],
                    constants.PREDICT_KEY: predictions,
                    constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                    constants.MARKER_KEY: features[constants.MARKER_KEY],
                    constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX]
                },
                vocab=all_targets[constants.LABEL_KEY],
                eval_tensor=f1_score, eval_update=tf.assign(f1_score, eval_placeholder),
                eval_placeholder=eval_placeholder)
            evaluation_hooks = [eval_hook]
        else:
            evaluation_hooks = [SequenceEvalHook(script_path=params.script_path,
                                                 tensors={
                                                     constants.LABEL_KEY: targets[constants.LABEL_KEY],
                                                     constants.PREDICT_KEY: predictions,
                                                     constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                                                     constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX]
                                                 },
                                                 vocab=all_targets[constants.LABEL_KEY],
                                                 output_file=params.output)]

    if mode == tf.estimator.ModeKeys.PREDICT:
        index_to_label = index_to_string_table_from_file(vocabulary_file=params.label_vocab_path,
                                                         default_value=all_targets[constants.LABEL_KEY].unknown_word)
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

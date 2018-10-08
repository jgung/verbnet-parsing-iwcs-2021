import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.saved_model import signature_constants

import tfnlp.common.constants as constants
from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.eval import ParserEvalHook, log_trainable_variables
from tfnlp.layers.layers import encoder, input_layer, numpy_orthogonal_matrix


def parser_model_func(features, mode, params):
    # (1) compute input layer, concatenating features (such as word embeddings) for each input token
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)

    # (2) transform inputs, producing a contextual vector for each input token
    outputs, output_size, _ = encoder(features, inputs, mode, params.config)

    input_shape = get_shape(outputs)  # (b x n x d), d == output_size
    n_steps = input_shape[1]  # n

    # (3) apply 2 arc and 2 rel MLPs to each output vector (1 for representing dependents, 1 for heads)
    def _mlp(size, name):
        return mlp(outputs, input_shape, params.config.mlp_dropout, size, mode == tf.estimator.ModeKeys.TRAIN, name, n_splits=2)

    arc_mlp_size, rel_mlp_size = 500, 100
    dep_arc_mlp, head_arc_mlp = _mlp(arc_mlp_size, name="arc_mlp")  # (bn x d), where d == arc_mlp_size
    dep_rel_mlp, head_rel_mlp = _mlp(rel_mlp_size, name="rel_mlp")  # (bn x d), where d == rel_mlp_size

    # (4) apply binary biaffine classifier for arcs
    with tf.variable_scope("arc_bilinear_logits"):
        arc_logits = bilinear(dep_arc_mlp, head_arc_mlp, 1, n_steps, include_bias2=False)  # (b x n x n)
        arc_predictions = tf.argmax(arc_logits, axis=-1)  # (b x n)

    # (5) apply variable class biaffine classifier for rels
    with tf.variable_scope("rel_bilinear_logits"):
        num_labels = params.extractor.targets[constants.DEPREL_KEY].vocab_size()  # r
        rel_logits = bilinear(dep_rel_mlp, head_rel_mlp, num_labels, n_steps)  # (b x n x r x n)

    arc_targets = None
    rel_targets = None
    loss = None  # combined arc + rel loss, used during both training evaluation
    mask = None

    # (6) compute combined arc and rel losses (both via softmax cross entropy)
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()

        mask = tf.sequence_mask(features[constants.LENGTH_KEY], name="padding_mask")

        def compute_loss(logits, targets, name):
            with tf.variable_scope(name):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
                losses = tf.boolean_mask(losses, mask)
                return tf.reduce_mean(losses)

        arc_targets = tf.identity(features[constants.HEAD_KEY], name=constants.HEAD_KEY)
        rel_targets = tf.identity(features[constants.DEPREL_KEY], name=constants.DEPREL_KEY)

        arc_loss = compute_loss(arc_logits, arc_targets, "arc_bilinear_loss")
        _rel_logits = select_logits(rel_logits, arc_targets, n_steps)
        rel_loss = compute_loss(_rel_logits, rel_targets, "rel_bilinear_loss")

        loss = arc_loss + rel_loss

    # (7) compute gradients w.r.t. trainable network parameters and apply update using optimization algorithm
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    arc_probs = None
    rel_probs = None
    n_tokens = None
    rel_predictions = None

    # (8) compute relations, and arc/prob probabilities for use in MST algorithm
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        arc_probs = tf.nn.softmax(arc_logits)  # (b x n)
        rel_probs = tf.nn.softmax(rel_logits, axis=2)  # (b x n x r x n)
        n_tokens = tf.cast(tf.reduce_sum(features[constants.LENGTH_KEY]), tf.int32)
        _rel_logits = select_logits(rel_logits, arc_predictions, n_steps)  # (b x n x r)
        rel_predictions = tf.argmax(_rel_logits, axis=-1)  # (b x n)

    eval_metric_ops = None
    evaluation_hooks = None

    # (9) compute metrics, such as UAS, LAS, and LA
    if mode == tf.estimator.ModeKeys.EVAL:
        arc_correct = tf.boolean_mask(tf.to_int32(tf.equal(arc_predictions, arc_targets)), mask)
        rel_correct = tf.boolean_mask(tf.to_int32(tf.equal(rel_predictions, rel_targets)), mask)
        n_arc_correct = tf.cast(tf.reduce_sum(arc_correct), tf.int32)
        n_rel_correct = tf.cast(tf.reduce_sum(rel_correct), tf.int32)
        correct = arc_correct * rel_correct
        n_correct = tf.cast(tf.reduce_sum(correct), tf.int32)

        eval_metric_ops = {
            constants.UNLABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_arc_correct / n_tokens),
            constants.LABEL_SCORE: tf.metrics.mean(n_rel_correct / n_tokens),
            constants.LABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_correct / n_tokens)
        }

        evaluation_hooks = []

        if params.script_path:
            hook = ParserEvalHook(
                {
                    constants.ARC_PROBS: arc_probs,
                    constants.REL_PROBS: rel_probs,
                    constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                    constants.HEAD_KEY: features[constants.HEAD_KEY],
                    constants.DEPREL_KEY: features[constants.DEPREL_KEY]
                }, features=params.extractor, script_path=params.script_path)
            evaluation_hooks.append(hook)

    export_outputs = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: PredictOutput(arc_probs),
                          constants.DEPREL_KEY: PredictOutput(rel_probs)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions={
                                          constants.DEPREL_KEY: rel_predictions,
                                          constants.HEAD_KEY: arc_predictions,
                                          constants.ARC_PROBS: arc_probs,
                                          constants.REL_PROBS: rel_probs
                                      },
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)


def get_shape(tensor):
    shape = []
    dynamic_shape = tf.shape(tensor)
    static_shape = tensor.get_shape().as_list()
    for i in range(len(static_shape) - 1):
        shape.append(dynamic_shape[i])
    shape.append(static_shape[-1])
    return shape


def select_logits(logits, targets, n_steps):
    _one_hot = tf.one_hot(targets, n_steps)  # (b x n x n) one-hot Tensor
    # don't include gradients for arcs that aren't predicted
    # (b x n x r x n) (b x n x n x 1)
    # a.k.a. b*n (r x n) (n, 1) matrix multiplications
    _select_logits = tf.matmul(logits, tf.expand_dims(_one_hot, -1))  # (b x n x r x 1)
    _select_logits = tf.squeeze(_select_logits, axis=-1)  # (b x n x r)
    return _select_logits


def mlp(inputs, inputs_shape, dropout_rate, output_size, training, name, n_splits=1):
    def _leaky_relu(features):
        return tf.nn.leaky_relu(features, alpha=0.1)

    def _dropout(_inputs):
        size = _inputs.get_shape().as_list()[-1]
        dropped = tf.layers.dropout(_inputs, dropout_rate, noise_shape=[inputs_shape[0], 1, size], training=training)
        dropped = tf.reshape(dropped, [-1, size])
        return dropped

    inputs = _dropout(inputs)

    initializer = None
    if training and not tf.get_variable_scope().reuse:
        # initialize each split of the MLP into an individual orthonormal matrix
        mat = numpy_orthogonal_matrix([inputs_shape[-1], output_size])
        mat = np.concatenate([mat] * n_splits, axis=1)
        initializer = tf.constant_initializer(mat)

    output_size *= n_splits

    result = tf.layers.dense(inputs, output_size, activation=_leaky_relu, kernel_initializer=initializer, name=name)
    result = tf.reshape(result, [inputs_shape[0], inputs_shape[1], output_size])
    result = _dropout(result)

    if n_splits == 1:
        return result
    return tf.split(result, num_or_size_splits=n_splits, axis=1)


def bilinear(input1, input2, output_size, timesteps, include_bias1=True, include_bias2=True):
    """
    Compute bilinear attention as described in 'Deep Biaffine Attention for Neural Dependency Parsing' (Dozat and Manning, 2017).
    :param input1: (bn x d1) `Tensor`
    :param input2: (bn x d2) `Tensor`
    :param output_size: number of outputs
    :param timesteps: number of timesteps in input, n
    :param include_bias1: if `True`, make first transformation affine
    :param include_bias2: if `True`, add biases to both linear transformations
    :return: bilinear mapping (b x n x r x n) `Tensor` between `input1` and `input2`, or (b x n x n) if `output_size` == 1
    """

    def _add_bias(_input):
        batches_and_tokens = tf.shape(_input)[0]
        bias = tf.ones([batches_and_tokens, 1])
        return tf.concat([_input, bias], -1)

    if include_bias1:
        # (bn x d) -> (bn x d+1)
        input1 = _add_bias(input1)
    if include_bias2:
        input2 = _add_bias(input2)

    input1dim = input1.get_shape()[-1]
    input2dim = input2.get_shape()[-1]

    weights = tf.get_variable("weights", shape=[input1dim, output_size, input2dim], initializer=tf.orthogonal_initializer)

    # (d x r x d) -> (d x rd)
    weights = tf.reshape(weights, [input1dim, -1])
    # (b n x d) (d x r d) -> b n x r d
    linear_mapping = tf.matmul(input1, weights)
    # (bn x rd) -> (b x nr x d)
    linear_mapping = tf.reshape(linear_mapping, [-1, output_size * timesteps, input2dim])
    # (b x n x d)
    input2 = tf.reshape(input2, [-1, timesteps, input2dim])
    # (b x nr x d) (b x n x d)T -> (b x n x rn) (batch matrix multiplication)
    bilinear_result = tf.matmul(linear_mapping, input2, transpose_b=True)

    if output_size != 1:
        # (b x n x rn) -> (b x n x r x n)
        bilinear_result = tf.reshape(bilinear_result, [-1, timesteps, output_size, timesteps])

    return bilinear_result

import numpy as np
import tensorflow as tf

import tfnlp.common.constants as constants
from tfnlp.common.config import append_label
from tfnlp.common.eval_hooks import ParserEvalHook
from tfnlp.layers.heads import ModelHead
from tfnlp.layers.layers import numpy_orthogonal_matrix


class ParserHead(ModelHead):

    def __init__(self, inputs, config, features, params, training):
        super().__init__(inputs, config, features, params, training)
        self.arc_predictions = None
        self.arc_logits = None
        self.rel_logits = None
        self.n_steps = None

        self.arc_targets = None
        self.mask = None

        self.arc_probs = None
        self.rel_probs = None
        self.n_tokens = None
        self.predictions = None

    def _all(self):
        inputs = self.inputs[0]
        input_shape = get_shape(inputs)  # (b x n x d), d == output_size
        self.n_steps = input_shape[1]  # n

        # apply 2 arc and 2 rel MLPs to each output vector (1 for representing dependents, 1 for heads)
        def _mlp(size, name):
            return mlp(inputs, input_shape, self.config.mlp_dropout, size, self._training, name, n_splits=2)

        arc_mlp_size, rel_mlp_size = 500, 100
        dep_arc_mlp, head_arc_mlp = _mlp(arc_mlp_size, name="arc_mlp")  # (bn x d), where d == arc_mlp_size
        dep_rel_mlp, head_rel_mlp = _mlp(rel_mlp_size, name="rel_mlp")  # (bn x d), where d == rel_mlp_size

        # apply binary biaffine classifier for arcs
        with tf.variable_scope("arc_bilinear_logits"):
            self.arc_logits = bilinear(dep_arc_mlp, head_arc_mlp, 1, self.n_steps, include_bias2=False)  # (b x n x n)
            self.arc_predictions = tf.argmax(self.arc_logits, axis=-1)  # (b x n)

        # apply variable class biaffine classifier for rels
        with tf.variable_scope("rel_bilinear_logits"):
            num_labels = self.extractor.vocab_size()  # r
            self.rel_logits = bilinear(dep_rel_mlp, head_rel_mlp, num_labels, self.n_steps)  # (b x n x r x n)

    def _train_eval(self):
        # compute combined arc and rel losses (both via softmax cross entropy)

        self.mask = tf.sequence_mask(self.features[constants.LENGTH_KEY], name="padding_mask")

        def compute_loss(logits, targets, name):
            with tf.variable_scope(name):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
                losses = tf.boolean_mask(losses, self.mask)
                return tf.reduce_mean(losses)

        self.arc_targets = tf.identity(self.features[constants.HEAD_KEY], name=constants.HEAD_KEY)

        arc_loss = compute_loss(self.arc_logits, self.arc_targets, "arc_bilinear_loss")
        _rel_logits = select_logits(self.rel_logits, self.arc_targets, self.n_steps)
        rel_loss = compute_loss(_rel_logits, self.targets, "rel_bilinear_loss")

        self.loss = arc_loss + rel_loss
        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        # compute relations, and arc/prob probabilities for use in MST algorithm
        self.arc_probs = tf.nn.softmax(self.arc_logits)  # (b x n)
        self.rel_probs = tf.nn.softmax(self.rel_logits, axis=2)  # (b x n x r x n)
        self.n_tokens = tf.cast(tf.reduce_sum(self.features[constants.LENGTH_KEY]), tf.int32)
        _rel_logits = select_logits(self.rel_logits, self.arc_predictions, self.n_steps)  # (b x n x r)
        self.predictions = tf.argmax(_rel_logits, axis=-1)  # (b x n)

    def _evaluation(self):
        # compute metrics, such as UAS, LAS, and LA
        arc_correct = tf.boolean_mask(tf.to_int32(tf.equal(self.arc_predictions, self.arc_targets)), self.mask)
        rel_correct = tf.boolean_mask(tf.to_int32(tf.equal(self.predictions, self.targets)), self.mask)
        n_arc_correct = tf.cast(tf.reduce_sum(arc_correct), tf.int32)
        n_rel_correct = tf.cast(tf.reduce_sum(rel_correct), tf.int32)
        correct = arc_correct * rel_correct
        n_correct = tf.cast(tf.reduce_sum(correct), tf.int32)

        self.metric_ops = {
            constants.UNLABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_arc_correct / self.n_tokens),
            constants.LABEL_SCORE: tf.metrics.mean(n_rel_correct / self.n_tokens),
            constants.LABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_correct / self.n_tokens),
        }

        self.evaluation_hooks = []

        if self.params.script_path:
            hook = ParserEvalHook(
                {
                    constants.ARC_PROBS: self.arc_probs,
                    constants.REL_PROBS: self.rel_probs,
                    constants.LENGTH_KEY: self.features[constants.LENGTH_KEY],
                    constants.HEAD_KEY: self.features[constants.HEAD_KEY],
                    constants.DEPREL_KEY: self.features[constants.DEPREL_KEY]
                }, features=self.extractor, script_path=self.params.script_path)
            self.evaluation_hooks.append(hook)

    def _prediction(self):
        self.export_outputs = {constants.REL_PROBS: self.rel_probs,
                               constants.ARC_PROBS: self.arc_probs}


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

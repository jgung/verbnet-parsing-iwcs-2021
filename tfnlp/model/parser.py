import tensorflow as tf

import tfnlp.common.constants as constants
from tfnlp.common.config import append_label
from tfnlp.common.eval_hooks import ParserEvalHook
from tfnlp.layers.heads import ModelHead
from tfnlp.layers.util import select_logits, bilinear, get_shape, mlp


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

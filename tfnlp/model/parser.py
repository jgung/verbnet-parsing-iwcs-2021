import os

import tensorflow as tf

import tfnlp.common.constants as constants
from tfnlp.cli.evaluators import DepParserEvaluator
from tfnlp.common.config import append_label
from tfnlp.common.eval_hooks import ParserEvalHook
from tfnlp.layers.heads import ModelHead
from tfnlp.layers.layers import get_encoder_input
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
        self.predictions = None
        self.lens = self.features[constants.LENGTH_KEY] + 1  # plus one for sentinel

    def _all(self):
        inputs = get_encoder_input(self.inputs)
        input_shape = get_shape(inputs)  # (b x n x d), d == output_size
        self.n_steps = input_shape[1]  # n

        # apply 2 arc and 2 rel MLPs to each output vector (1 for representing dependents, 1 for heads)
        def _mlp(size, name):
            return mlp(inputs, input_shape, self.config.mlp_dropout, size, self._training, name, n_splits=2)

        dep_arc_mlp, head_arc_mlp = _mlp(self.config.arc_mlp_size, name="arc_mlp")  # (bn x d), where d == arc_mlp_size
        dep_rel_mlp, head_rel_mlp = _mlp(self.config.rel_mlp_size, name="rel_mlp")  # (bn x d), where d == rel_mlp_size

        # apply binary biaffine classifier for arcs
        with tf.variable_scope("arc_bilinear_logits"):
            self.arc_logits = bilinear(dep_arc_mlp, head_arc_mlp, 1, self.n_steps, include_bias2=False)  # (b x n x n)
            self.arc_predictions = tf.argmax(self.arc_logits, axis=-1)  # (b x n)

        # apply variable class biaffine classifier for rels
        with tf.variable_scope("rel_bilinear_logits"):
            num_labels = self.extractor.vocab_size()  # r
            self.rel_logits = bilinear(dep_rel_mlp, head_rel_mlp, num_labels, self.n_steps)  # (b x n x r x n)

    def _train_eval(self):
        self.mask = tf.sequence_mask(self.lens, name="padding_mask")

        # compute combined arc and rel losses (both via softmax cross entropy)
        def compute_loss(logits, targets, name):
            with tf.variable_scope(name):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
                losses = tf.boolean_mask(losses, self.mask)
                return tf.reduce_mean(losses)

        self.arc_targets = tf.identity(self.features[constants.HEAD_KEY], name=constants.HEAD_KEY)

        arc_loss = compute_loss(self.arc_logits, self.arc_targets, "arc_bilinear_loss")
        _rel_logits = select_logits(self.rel_logits, self.arc_targets, self.n_steps)
        rel_loss = compute_loss(_rel_logits, self.targets, "rel_bilinear_loss")

        arc_loss = self.config.get('arc_loss_weight', 1) * arc_loss
        rel_loss = self.config.get('rel_loss_weight', 1) * rel_loss
        self.loss = arc_loss + rel_loss
        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        # compute relations, and arc/prob probabilities for use in MST algorithm
        self.arc_probs = tf.nn.softmax(self.arc_logits)  # (b x n)
        self.rel_probs = tf.nn.softmax(self.rel_logits, axis=2)  # (b x n x r x n)
        _rel_logits = select_logits(self.rel_logits, self.arc_predictions, self.n_steps)  # (b x n x r)
        self.predictions = tf.argmax(_rel_logits, axis=-1)  # (b x n)

    def _evaluation(self):
        # compute metrics, such as UAS, LAS, and LA
        arc_correct = tf.boolean_mask(tf.to_int32(tf.equal(self.arc_predictions[:, 1:], self.arc_targets[:, 1:])),
                                      self.mask[:, 1:])
        rel_correct = tf.boolean_mask(tf.to_int32(tf.equal(self.predictions[:, 1:], self.targets[:, 1:])),
                                      self.mask[:, 1:])
        n_arc_correct = tf.cast(tf.reduce_sum(arc_correct), tf.int32)
        n_rel_correct = tf.cast(tf.reduce_sum(rel_correct), tf.int32)
        correct = arc_correct * rel_correct
        n_correct = tf.cast(tf.reduce_sum(correct), tf.int32)

        n_tokens = tf.cast(tf.reduce_sum(self.lens - 1), tf.int32)  # minus 1 for sentinel
        self.metric_ops = {
            constants.UNLABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_arc_correct / n_tokens),
            constants.LABEL_SCORE: tf.metrics.mean(n_rel_correct / n_tokens),
            constants.LABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_correct / n_tokens),
        }

        overall_score = tf.identity(self.metric)
        self.metric_ops[append_label(constants.OVERALL_KEY, self.name)] = (overall_score, overall_score)
        overall_key = append_label(constants.OVERALL_KEY, self.name)
        # https://github.com/tensorflow/tensorflow/issues/20418 -- metrics don't accept variables, so we create a tensor
        eval_placeholder = tf.placeholder(dtype=tf.float32, name='update_%s' % overall_key)

        self.evaluation_hooks = []

        hook = ParserEvalHook(
            {
                constants.ARC_PROBS: self.arc_probs,
                constants.REL_PROBS: self.rel_probs,
                constants.LENGTH_KEY: self.lens,  # plus one for the sentinel
                constants.HEAD_KEY: self.features[constants.HEAD_KEY],
                constants.DEPREL_KEY: self.features[constants.DEPREL_KEY],
                constants.SENTENCE_INDEX: self.features[constants.SENTENCE_INDEX]
            },
            evaluator=DepParserEvaluator(
                target=self.extractor,
                output_path=os.path.join(self.params.job_dir, self.name + '.dev'),
                script_path=self.params.script_path
            ),
            eval_update=tf.assign(self.metric, eval_placeholder),
            eval_placeholder=eval_placeholder,
            output_dir=self.params.job_dir
        )
        self.evaluation_hooks.append(hook)

    def _prediction(self):
        self.export_outputs = {constants.REL_PROBS: self.rel_probs,
                               constants.ARC_PROBS: self.arc_probs}

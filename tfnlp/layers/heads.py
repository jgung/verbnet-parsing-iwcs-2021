import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf.python.ops import crf
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.config import append_label
from tfnlp.common.eval_hooks import ClassifierEvalHook, SequenceEvalHook, SrlEvalHook
from tfnlp.common.metrics import tagger_metrics
from tfnlp.layers.layers import string2index
from tfnlp.layers.util import get_shape, mlp, bilinear, select_logits, sequence_loss


class ModelHead(object):
    def __init__(self, inputs, config, features, params, training):
        self.inputs = inputs
        self.config = config
        self.name = config.name
        self.extractor = params.extractor.targets[self.name]
        self.features = features
        self.params = params
        self._training = training

        self.targets = None
        self.logits = None
        self.loss = None
        self.predictions = None
        self.evaluation_hooks = []
        self.metric_ops = {}
        self.metric = None
        self.export_outputs = {}

    def training(self):
        self.targets = self.features[self.name]
        if self.extractor.has_vocab():
            self.targets = string2index(self.features[self.name], self.extractor)

        with tf.variable_scope(self.name):
            self._all()
            self._train_eval()
            self._train()

    def evaluation(self):
        self.targets = self.features[self.name]
        if self.extractor.has_vocab():
            self.targets = string2index(self.features[self.name], self.extractor)

        with tf.variable_scope(self.name):
            self._all()
            self._train_eval()
            self._eval_predict()
            self._evaluation()

    def prediction(self):
        with tf.variable_scope(self.name):
            self._all()
            self._eval_predict()
            self._prediction()

    def _all(self):
        """
        Called for every setting (training/evaluation/prediction).
        """
        pass

    def _train_eval(self):
        """
        Called after `_all` for training/evaluation.
        """
        pass

    def _train(self):
        """
        Called after `_train_eval` for training.
        """
        pass

    def _eval_predict(self):
        """
        Called after `_train_eval` for evaluation and after `_all` for prediction.
        """
        pass

    def _evaluation(self):
        """
        Called after `_eval_predict` for evaluation.
        """
        pass

    def _prediction(self):
        """
        Called after `_eval_predict` for prediction.
        """
        index_to_label = index_to_string_table_from_file(vocabulary_file=os.path.join(self.params.vocab_path, self.name),
                                                         default_value=self.extractor.unknown_word)
        self.predictions = tf.identity(index_to_label.lookup(tf.cast(self.predictions, dtype=tf.int64)), name="labels")
        self.export_outputs = {self.name: self.predictions}


class ClassifierHead(ModelHead):
    def __init__(self, inputs, config, features, params, training):
        super().__init__(inputs, config, features, params, training)
        self._sequence_lengths = self.features[constants.LENGTH_KEY]
        self.scores = None

    def _all(self):
        inputs = self.inputs[2]
        inputs = tf.layers.dropout(inputs, training=self._training)

        with tf.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            self.logits = tf.layers.dense(inputs, num_labels, kernel_initializer=tf.zeros_initializer)

    def _train_eval(self):
        if self.config.label_smoothing > 0:
            targets = tf.one_hot(self.targets, depth=self.extractor.vocab_size())
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=targets,
                                                        logits=self.logits,
                                                        label_smoothing=self.config.label_smoothing)
        else:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

    def _eval_predict(self):
        self.scores = tf.nn.softmax(self.logits)  # (b x n)
        self.predictions = tf.argmax(self.logits, axis=1)

    def training(self):
        super().training()
        self._eval_predict()
        self._prediction()

    def evaluation(self):
        super().evaluation()
        self._prediction()

    def _evaluation(self):
        predictions_key = append_label(constants.PREDICT_KEY, self.name)
        labels_key = append_label(constants.LABEL_KEY, self.name)
        acc_key = append_label(constants.ACCURACY_METRIC_KEY, self.name)

        self.metric_ops = {acc_key: tf.metrics.accuracy(labels=self.targets, predictions=self.predictions, name=acc_key)}

        tensors = {
            labels_key: self.targets,
            predictions_key: self.predictions,
            constants.LABEL_SCORES: self.scores,
            constants.LENGTH_KEY: self._sequence_lengths,
            constants.SENTENCE_INDEX: self.features[constants.SENTENCE_INDEX],
        }

        constraint_key = self.extractor.constraint_key
        if constraint_key:
            tensors[constraint_key] = self.features[constraint_key]

        self.evaluation_hooks = [
            ClassifierEvalHook(
                label_key=labels_key,
                predict_key=predictions_key,
                tensors=tensors,
                vocab=self.extractor,
                output_dir=self.params.job_dir
            )
        ]


def select_by_token_index(states, indices):
    row_indices = tf.range(tf.shape(indices, out_type=tf.int64)[0])
    full_indices = tf.stack([row_indices, indices], axis=1)
    return tf.gather_nd(states, indices=full_indices)


class TokenClassifierHead(ClassifierHead):
    def __init__(self, inputs, config, features, params, training):
        super().__init__(inputs, config, features, params, training)

    def _all(self):
        if isinstance(self.inputs, tf.Tensor):
            inputs = self.inputs
        else:
            inputs = self.inputs[0]
        if constants.TOKEN_INDEX_KEY in self.features:
            targets = self.features[constants.TOKEN_INDEX_KEY]
        else:
            targets = self.features[constants.PREDICATE_INDEX_KEY]
        inputs = select_by_token_index(inputs, targets)

        with tf.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            self.logits = tf.layers.dense(inputs, num_labels, kernel_initializer=tf.zeros_initializer)


def create_transition_matrix(labels):
    """
    Return a numpy matrix to enforce valid transitions for IOB-style tagging problems.
    :param labels: label feature extractor
    """
    labels = [labels.index_to_feat(i) for i in range(len(labels.indices))]
    num_tags = len(labels)
    transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
    for i, prev_label in enumerate(labels):
        for j, curr_label in enumerate(labels):
            if curr_label == BERT_SUBLABEL:
                transition_params[i, j] = np.NINF
            elif i == j:
                continue
            elif curr_label[:2] == 'I-' and prev_label != 'B-' + curr_label[2:]:
                transition_params[i, j] = np.NINF
    return tf.initializers.constant(transition_params)


class TaggerHead(ModelHead):

    def __init__(self, inputs, config, features, params, training=False):
        super().__init__(inputs, config, features, params, training)
        if constants.BERT_SPLIT_INDEX in self.features:
            self._sequence_lengths = self.features[constants.BERT_SPLIT_INDEX]
        else:
            self._sequence_lengths = self.features[constants.LENGTH_KEY]

        self._tag_transitions = None

    def training(self):
        super().training()
        self._eval_predict()
        self._prediction()

    def evaluation(self):
        super().evaluation()
        self._prediction()

    def _all(self):
        inputs, encoder_dim = self.inputs[:2]
        time_steps = tf.shape(inputs)[1]

        # flatten encoder outputs to a (batch_size * time_steps x encoder_dim) Tensor for batch matrix multiplication
        inputs = tf.reshape(inputs, [-1, encoder_dim], name="flatten")

        with tf.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            initializer = tf.zeros_initializer if self.config.zero_init else tf.random_normal_initializer(stddev=0.01)

            dense = tf.layers.dense(inputs, num_labels, kernel_initializer=initializer)
            # batch multiplication complete, convert back to a (batch_size x time_steps x num_labels) Tensor
            self.logits = tf.reshape(dense, [-1, time_steps, num_labels], name="unflatten")
        if self.config.crf:
            # explicitly train a transition matrix
            self._tag_transitions = tf.get_variable("transitions", [num_labels, num_labels])
        else:
            # use constrained decoding based on IOB labels
            self._tag_transitions = tf.get_variable("transitions", [num_labels, num_labels], trainable=False,
                                                    initializer=create_transition_matrix(self.extractor))

    def _train_eval(self):
        num_labels = self.extractor.vocab_size()
        self.loss = sequence_loss(logits=self.logits,
                                  targets=self.targets,
                                  sequence_lengths=self._sequence_lengths,
                                  num_labels=num_labels,
                                  crf=self.config.crf, tag_transitions=self._tag_transitions,
                                  label_smoothing=self.config.label_smoothing,
                                  confidence_penalty=self.config.confidence_penalty,
                                  mask=self.features.get(constants.SEQUENCE_MASK))

        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        predictions = crf.crf_decode(self.logits, self._tag_transitions, tf.cast(self._sequence_lengths, tf.int32))[0]
        # optionally mask intermediate subtokens from prediction results
        self.predictions = self._mask_subtokens(predictions)

    def _evaluation(self):
        self.evaluation_hooks = []
        self.metric_ops = {}
        predictions_key = append_label(constants.PREDICT_KEY, self.name)
        labels_key = append_label(constants.LABEL_KEY, self.name)

        eval_tensors = {  # tensors necessary for evaluation hooks (such as sequence length)
            constants.LENGTH_KEY: self._sequence_lengths,
            constants.SENTENCE_INDEX: self.features[constants.SENTENCE_INDEX],
            labels_key: self._mask_subtokens(self.targets),
            predictions_key: self.predictions,
        }

        overall_score = tf.identity(self.metric)
        self.metric_ops[append_label(constants.OVERALL_KEY, self.name)] = (overall_score, overall_score)
        overall_key = append_label(constants.OVERALL_KEY, self.name)
        # https://github.com/tensorflow/tensorflow/issues/20418 -- metrics don't accept variables, so we create a tensor
        eval_placeholder = tf.placeholder(dtype=tf.float32, name='update_%s' % overall_key)

        if constants.SRL_KEY in self.config.task:
            eval_tensors[constants.MARKER_KEY] = self.features[constants.MARKER_KEY]

            self.evaluation_hooks.append(
                SrlEvalHook(
                    tensors=eval_tensors,
                    vocab=self.extractor,
                    label_key=labels_key,
                    predict_key=predictions_key,
                    eval_update=tf.assign(self.metric, eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    output_confusions=self.params.verbose_eval,
                    output_dir=self.params.job_dir
                )
            )
        else:
            ns = None if self.name == constants.LABEL_KEY else self.name
            metrics = tagger_metrics(labels=self.targets, predictions=tf.cast(self.predictions, dtype=tf.int64), ns=ns)
            self.metric_ops.update(metrics)
            acc_key = append_label(constants.ACCURACY_METRIC_KEY, self.name)
            self.metric_ops[acc_key] = tf.metrics.accuracy(labels=self.targets, predictions=self.predictions, name=acc_key)

            self.evaluation_hooks.append(
                SequenceEvalHook(
                    tensors=eval_tensors,
                    vocab=self.extractor,
                    label_key=labels_key,
                    predict_key=predictions_key,
                    eval_update=tf.assign(self.metric, eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    output_dir=self.params.job_dir
                )
            )

    def _mask_subtokens(self, tensor_with_subtokens):
        mask = self.features.get(constants.SEQUENCE_MASK)
        if mask is not None:
            cond = tf.greater(mask, tf.zeros(tf.shape(mask), tf.int64))
            ignore = self.extractor.feat2index(BERT_SUBLABEL)
            tensor_with_subtokens = tf.where(cond,
                                             tf.cast(tensor_with_subtokens, tf.int64),
                                             tf.cast(tf.fill(tf.shape(tensor_with_subtokens), ignore), tf.int64))
            return tensor_with_subtokens
        return tensor_with_subtokens


class BiaffineSrlHead(TaggerHead):

    def __init__(self, inputs, config, features, params, training=False):
        super().__init__(inputs, config, features, params, training)
        self.n_steps = None
        self.predicate_indices = None

    def _all(self):
        inputs, encoder_dim = self.inputs[:2]

        input_shape = get_shape(inputs)  # (b x n x d), d == output_size
        self.n_steps = input_shape[1]  # n

        # apply 2 arc and 2 rel MLPs to each output vector (1 for representing dependents, 1 for heads)
        def _mlp(size, name):
            return mlp(inputs, input_shape, self.config.mlp_dropout, size, self._training, name, n_splits=2)

        arg_mlp, predicate_mlp = _mlp(self.config.mlp_dim, name="rel_mlp")  # (bn x d), where d == rel_mlp_size

        # apply variable class biaffine classifier for semantic role labels
        with tf.variable_scope("bilinear_logits"):
            num_labels = self.extractor.vocab_size()  # r
            initializer = tf.zeros_initializer if self.config.zero_init else None
            self.logits = bilinear(arg_mlp, predicate_mlp, num_labels, self.n_steps, initializer=initializer)  # (b x n x r x n)

        if self.config.crf:
            # explicitly train a transition matrix
            self._tag_transitions = tf.get_variable("transitions", [num_labels, num_labels])
        else:
            # use constrained decoding based on IOB labels
            self._tag_transitions = tf.get_variable("transitions", [num_labels, num_labels], trainable=False,
                                                    initializer=create_transition_matrix(self.extractor))

        # batch-length vector of predicate indices
        predicate_indices = self.features[constants.PREDICATE_INDEX_KEY]
        predicate_indices = tf.expand_dims(predicate_indices, -1)
        # convert to [batch x n_steps] size Tensor, since each token's head is the predicate
        self.predicate_indices = tf.tile(predicate_indices, [1, self.n_steps])

    def _train_eval(self):
        self.mask = tf.sequence_mask(self.features[constants.LENGTH_KEY], name="padding_mask")

        num_labels = self.extractor.vocab_size()
        _logits = select_logits(self.logits, self.predicate_indices, self.n_steps)

        rel_loss = sequence_loss(logits=_logits,
                                 targets=self.targets,
                                 sequence_lengths=self._sequence_lengths,
                                 num_labels=num_labels,
                                 crf=self.config.crf,
                                 tag_transitions=self._tag_transitions,
                                 label_smoothing=self.config.label_smoothing,
                                 confidence_penalty=self.config.confidence_penalty, name="bilinear_loss",
                                 mask=self.features.get(constants.SEQUENCE_MASK))

        self.loss = rel_loss
        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        self.rel_probs = tf.nn.softmax(self.logits, axis=2)  # (b x n x r x n)
        self.n_tokens = tf.cast(tf.reduce_sum(self.features[constants.LENGTH_KEY]), tf.int32)
        _logits = select_logits(self.logits, self.predicate_indices, self.n_steps)  # (b x n x r)
        self.predictions = crf.crf_decode(_logits, self._tag_transitions, tf.cast(self._sequence_lengths, tf.int32))[0]

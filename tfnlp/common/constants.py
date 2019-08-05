PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<BOS>"
END_WORD = "<EOS>"

PAD_INDEX = 0
UNKNOWN_INDEX = 1
START_INDEX = 2
END_INDEX = 3

LABEL_KEY = "gold"
LABEL_SCORES = "scores"
PREDICT_KEY = "pred"
WORD_KEY = "word"
POS_KEY = "pos"
CHUNK_KEY = "chunk"
NAMED_ENTITY_KEY = "ne"
SENSE_KEY = "sense"
LENGTH_KEY = "len"
CHAR_KEY = "char"
ELMO_KEY = "elmo"
BERT_KEY = "bert"
BERT_LENGTH_KEY = "bert_len"
SEQUENCE_MASK = "sequence_mask"
BERT_SPLIT_INDEX = "bert_split_idx"

ID_KEY = "ID"
LEMMA_KEY = "lemma"
PLEMMA_KEY = "plemma"
PPOS_KEY = "ppos"
FEAT_KEY = "feat"
PFEAT_KEY = "pfeat"
HEAD_KEY = "head"
PHEAD_KEY = "phead"
DEPREL_KEY = "deprel"
PDEPREL_KEY = "pdeprel"
FILLPRED_KEY = "fillpred"
PRED_KEY = "pred"
APREDs_KEY = "apred"
REL_PROBS = "rel_probs"
ARC_PROBS = "arc_probs"

XPOS_KEY = "xpos"
DEP_FEATS_KEY = "feats"
ENHANCED_DEPS_KEY = "edeps"
MISC_KEY = "misc"

MARKER_KEY = "marker"
TOKEN_INDEX_KEY = "token_index"
PREDICATE_FORM = "predicate_form"
PREDICATE_LEMMA = "predicate_lemma"
PREDICATE_INDEX_KEY = "predicate_index"
INSTANCE_INDEX = "instance_idx"
SENTENCE_INDEX = "sentence_idx"
PREDICATE_KEY = "predicate"
PARSE_KEY = "parse"

BEGIN = "B"
BEGIN_ = "B-"
END = "E"
END_ = "E-"
SINGLE = "S"
SINGLE_ = "S-"
IN = "I"
IN_ = "I-"
OUT = "O"
CONLL_START = "("
CONLL_CONT = "*"
CONLL_END = ")"

KEY_FIELD = "key"
NAME_FIELD = "name"
CONFIG_FIELD = "config"
INITIALIZER = "initializer"
INCLUDE_IN_VOCAB = "include_in_vocab"

OVERALL_KEY = 'Overall'
F1_METRIC_KEY = "F-Measure"
SRL_METRIC_KEY = 'F1-SRL'
RECALL_METRIC_KEY = "Recall"
PRECISION_METRIC_KEY = "Precision"
ACCURACY_METRIC_KEY = "Accuracy"

LABELED_ATTACHMENT_SCORE = "LAS"
UNLABELED_ATTACHMENT_SCORE = "UAS"
LABEL_SCORE = "LS"

TOKEN_CLASSIFIER_KEY = "token-classifier"
CLASSIFIER_KEY = "classifier"
TAGGER_KEY = "tagger"
NER_KEY = "ner"
SRL_KEY = "srl"
BIAFFINE_SRL_KEY = "biaffine-srl"
PARSER_KEY = "parser"

# encoder types, that apply a functon to one or more inputs
ENCODER_SENTINEL = 'sentinel'  # add a sentinel (head) to input
ENCODER_BLSTM = 'lstm'
ENCODER_DBLSTM = 'dblstm'
ENCODER_TRANSFORMER = 'transformer'
ENCODER_CONCAT = 'concat'
ENCODER_SUM = 'sum'
ENCODER_IDENT = 'identity'
ENCODER_MLP = 'mlp'
ENCODER_REPEAT = 'repeat_token'
ENCODER_REPEAT_AND_CONCAT = 'repeat_and_concat'

ENCODER_REMOVE_SUBTOKENS = 'remove_subtokens'
ENCODERS = [ENCODER_BLSTM, ENCODER_DBLSTM, ENCODER_TRANSFORMER, ENCODER_CONCAT, ENCODER_SUM, ENCODER_IDENT, ENCODER_MLP,
            ENCODER_REPEAT, ENCODER_REPEAT_AND_CONCAT, ENCODER_SENTINEL, ENCODER_REMOVE_SUBTOKENS]

# TF serving export constants
SERVING_PLACEHOLDER = 'input_example_tensor'

# paths/files used in trainer job directory
VOCAB_PATH = 'vocab'
CONFIG_PATH = 'config.json'
MODEL_PATH = 'model'

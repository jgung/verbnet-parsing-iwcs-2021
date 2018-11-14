PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<BOS>"
END_WORD = "<EOS>"

PAD_INDEX = 0
UNKNOWN_INDEX = 1
START_INDEX = 2
END_INDEX = 3

LABEL_KEY = "gold"
PREDICT_KEY = "pred"
WORD_KEY = "word"
POS_KEY = "pos"
CHUNK_KEY = "chunk"
NAMED_ENTITY_KEY = "ne"
LENGTH_KEY = "len"
CHAR_KEY = "char"
ELMO_KEY = "elmo"

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
INSTANCE_INDEX = "instance_idx"
SENTENCE_INDEX = "sentence_idx"
ROLESET_KEY = "roleset"
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
PARSER_KEY = "parser"

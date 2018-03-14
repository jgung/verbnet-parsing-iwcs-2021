from tfnlp.feature import Feature, FeatureExtractor, LengthFeature, SequenceFeature, SequenceListFeature
from tfnlp.layers.reduce import ConvNet


def get_reduce_function(func, dim, length):
    if func.name == "ConvNet":
        return ConvNet(input_size=dim, kernel_size=func.kernel_size, num_filters=func.num_filters, max_length=length)
    else:
        raise AssertionError("Unexpected feature function: {}".format(func.name))


def get_feature(feature):
    if feature.rank == 3:
        feat = SequenceListFeature
        feature.config.func = get_reduce_function(feature.config.function, feature.config.dim, feature.max_len)
    elif feature.rank == 2:
        feat = SequenceFeature
    elif feature.rank == 1:
        feat = Feature
    else:
        raise AssertionError("Unexpected feature rank: {}".format(feature.rank))
    return feat(**feature)


def get_feature_extractor(config):
    features = []
    for feature in config.features:
        features.append(get_feature(feature))
    targets = []
    for target in config.targets:
        targets.append(get_feature(target))

    features.append(LengthFeature(config.seq_feat))

    return FeatureExtractor(features, targets)

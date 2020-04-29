# noinspection PyProtectedMember
from tensorflow_estimator.python.estimator.exporter import BestExporter, _SavedModelExporter


# Temporary fix for https://github.com/tensorflow/tensorflow/issues/28570
class BesterExporter(BestExporter):
    def __init__(self, serving_input_receiver_fn=None, compare_fn=None, exports_to_keep=5):
        super().__init__(serving_input_receiver_fn=serving_input_receiver_fn,
                         compare_fn=compare_fn,
                         exports_to_keep=exports_to_keep)
        self._saved_model_exporter = _SavedModelExporter('best_exporter',
                                                         serving_input_receiver_fn=serving_input_receiver_fn,
                                                         assets_extra=None,
                                                         as_text=False)

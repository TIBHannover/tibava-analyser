from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import Annotation, AnnotationData, ShotsData, ScalarData, ListData, generate_id

import numpy as np


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_type_classifier",
}

default_parameters = {"threshold": 0.0}

requires = {
    "shots": ShotsData,
    "probs": ListData,
}

provides = {
    "annos": AnnotationData,
}


@AnalyserPluginManager.export("shot_annotator")
class ShotAnnotator(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def mean_shot_probabilities(self, start: float, end: float, probs: list):
        mean_class_probs = {}
        for label, class_probs in zip(probs.index, probs.data):
            idx = []
            for i, t in enumerate(class_probs.time):
                if t > start and t < end:
                    idx.append(i)

            if len(idx) > 0:
                mean_class_probs[label] = np.mean(class_probs.y[idx])
            else:
                mean_class_probs[label] = None

        return mean_class_probs

    def call(self, inputs, parameters, callbacks=None):
        annotations = []

        for i, shot in enumerate(inputs["shots"].shots):
            mean_class_probs = self.mean_shot_probabilities(start=shot.start, end=shot.end, probs=inputs["probs"])

            max_mean_class_prob = parameters.get("threshold")
            max_label = None
            for label, class_prob in mean_class_probs.items():
                if not class_prob:
                    continue

                if class_prob > max_mean_class_prob:
                    max_mean_class_prob = class_prob
                    max_label = label

            if max_label:
                annotations.append(
                    Annotation(start=shot.start, end=shot.end, labels=[max_label])
                )  # Maybe store max_mean_class_prob as well?
            self.update_callbacks(callbacks, progress=i / len(inputs["shots"].shots))

        self.update_callbacks(callbacks, progress=1.0)
        return {"annotations": AnnotationData(annotations=annotations)}

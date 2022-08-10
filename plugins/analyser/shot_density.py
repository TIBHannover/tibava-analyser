from analyser.plugins.manager import AnalyserPluginManager
from analyser.data import Annotation, AnnotationData, ShotsData, ScalarData, ListData, generate_id
from analyser.plugins import Plugin

import math
import numpy as np
from sklearn.neighbors import KernelDensity

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"kernel": "gaussian", "bandwidth": 30.0, "fps": 10}

requires = {
    "shots": ShotsData,
}

provides = {
    "shot_density": ScalarData,
}


@AnalyserPluginManager.export("shot_density")
class ShotDensity(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def call(self, inputs, parameters, callbacks=None):
        print(inputs)
        last_shot_end = 0
        shots = []
        for i, shot in enumerate(inputs["shots"].shots):
            shots.append(shot.start)

            if shot.end > last_shot_end:
                last_shot_end = shot.end

            self.update_callbacks(callbacks, progress=i / len(inputs["shots"].shots))

        time = np.linspace(0, last_shot_end, math.ceil(last_shot_end * parameters.get("fps")) + 1)[:, np.newaxis]
        shots = np.asarray(shots).reshape(-1, 1)
        kde = KernelDensity(kernel=parameters.get("kernel"), bandwidth=parameters.get("bandwidth")).fit(shots)
        log_dens = kde.score_samples(time)
        shot_density = np.exp(log_dens)
        shot_density = (shot_density - shot_density.min()) / (shot_density.max() - shot_density.min())

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "shot_density": ScalarData(
                y=shot_density.squeeze(), time=time.squeeze().tolist(), delta_time=1 / parameters.get("fps")
            )
        }

from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder, image_pad
from analyser.data import ListData, ScalarData, VideoData, ListData, generate_id
from analyser.plugins import Plugin
import redisai as rai
import ml2rt
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_type_classifier",
    "model_device": "cpu",
    "model_file": "/models/shot_type_classification/shot_type_classifier_e9-s3199_cpu.pt",
    "image_resolution": 224,
}

default_parameters = {"fps": 5}

requires = {
    "video": VideoData,
}

provides = {
    "probs": ListData,
}


@AnalyserPluginManager.export("shot_type_classifier")
class ShotTypeClassifier(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.image_resolution = self.config["image_resolution"]

        self.con = rai.Client(host=self.host, port=self.port)

        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="torch",
            device=self.model_device,
            data=model,
            batch=16,
        )

    def call(self, inputs, parameters):
        video_decoder = VideoDecoder(
            inputs["video"].path, max_dimension=self.image_resolution, fps=parameters.get("fps")
        )

        # video_decoder.fps
        job_id = generate_id()

        predictions = []
        time = []
        for i, frame in enumerate(video_decoder):
            frame = image_pad(frame["frame"])
            self.con.tensorset(f"data_{job_id}", frame)

            _ = self.con.modelrun(self.model_name, f"data_{job_id}", f"prob_{job_id}")
            prediction = self.con.tensorget(f"prob_{job_id}")
            predictions.append(prediction.tolist())
            time.append(i / parameters.get("fps"))
        # predictions = zip(*predictions)
        probs = ListData(data=[ScalarData(y=y, time=time) for y in zip(*predictions)])
        print(len(probs.data))

        # predictions: list(np.array) in form of [(p_ECU, p_CU, p_MS, p_FS, p_LS), ...] * #frames
        # times: list in form [0 / fps, 1 / fps, ..., #frames/fps]
        return {"probs": probs}

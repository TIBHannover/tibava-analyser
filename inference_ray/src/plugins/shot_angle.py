from analyser.inference.plugin import AnalyserPlugin, AnalyserPluginManager
from analyser.data import Annotation, AnnotationData, ShotsData, VideoData

from analyser.data import DataManager, Data
from analyser.utils import VideoDecoder

import logging
from typing import Callable, Dict


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "shot_angle_classifier",
}

default_parameters = {
    "fps": 2,
    "batch_size": 32,
}

requires = {
    "video": VideoData,
    "shots": ShotsData,
}

provides = {
    "annotations": AnnotationData,
}


@AnalyserPluginManager.export("shot_angle")
class ShotAngle(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import numpy as np
        import torch
        from torchvision.transforms import v2
        from transformers import AutoModelForImageClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"

        transform = v2.Compose(
            [
                v2.Resize(384, antialias=True),
                v2.CenterCrop((384, 384)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        model = AutoModelForImageClassification.from_pretrained(
            "gullalc/convnextv2-base-22k-384-cinescale-angle"
        )
        model.to(device)
        model.eval()

        with inputs["video"] as video_data, inputs["shots"] as shot_data:
            with video_data.open_video() as f_video, data_manager.create_data(
                "AnnotationData"
            ) as annotation_data:
                video_decoder = VideoDecoder(
                    path=f_video, extension=f".{video_data.ext}", fps=parameters["fps"]
                )

                for j, shot in enumerate(shot_data.shots):
                    start_index = int(video_decoder.fps() * shot.start)
                    end_index = int(video_decoder.fps() * shot.end)

                    shot_preds = []
                    _batch = []
                    for i, _frame in enumerate(video_decoder):
                        _batch.append(_frame.get("frame"))
                        if (
                            i + 1 == parameters["batch_size"]
                            or i >= end_index - start_index
                        ):
                            batch = torch.from_numpy(np.stack(_batch, axis=0))
                            batch = batch.permute((0, 3, 1, 2))
                            inputs = transform(batch).to(device)
                            with torch.no_grad():
                                outputs = model(inputs).logits

                            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                            shot_preds.extend(model.config.id2label[p] for p in preds)
                            _batch = []
                        if i >= end_index - start_index:
                            break

                    annotation_data.annotations.append(
                        Annotation(start=shot.start, end=shot.end, labels=shot_preds)
                    )
                    self.update_callbacks(callbacks, progress=j / len(shot_data.shots))

        self.update_callbacks(callbacks, progress=1.0)
        return {"annotations": annotation_data}

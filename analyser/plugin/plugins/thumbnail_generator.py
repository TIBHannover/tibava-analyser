import imageio

from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import ImageData, VideoData, ImagesData
from analyser.data import generate_id, create_data_path
from analyser.utils import VideoDecoder


default_config = {"data_dir": "/data"}


default_parameters = {"fps": 5.0, "max_dimension": 128}

requires = {
    "video": VideoData,
}

provides = {
    "images": ImageData,
}


@AnalyserPluginManager.export("thumbnail_generator")
class ThumbnailGenerator(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, inputs, parameters, callbacks=None):

        video_decoder = VideoDecoder(
            path=inputs["video"].path, fps=parameters.get("fps"), max_dimension=parameters.get("max_dimension")
        )

        images = []
        num_frames = video_decoder.duration() * video_decoder.fps()
        for i, frame in enumerate(video_decoder):

            self.update_callbacks(callbacks, progress=i / num_frames)
            image_id = generate_id()
            output_path = create_data_path(self.config.get("data_dir"), image_id, "jpg")
            imageio.imwrite(output_path, frame.get("frame"))
            images.append(
                ImageData(id=image_id, ext="jpg", time=frame.get("time"), delta_time=1 / parameters.get("fps"))
            )
        data = ImagesData(images=images)

        self.update_callbacks(callbacks, progress=1.0)
        return {"images": data}
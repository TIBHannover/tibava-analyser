import imageio.v3 as iio
import ffmpeg
import numpy as np


def parse_meta_ffmpeg(path):
    try:
        meta = ffmpeg.probe(path)

        streams = meta.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":

                duration = s.get("duration", None)
                try:
                    duration = float(duration)
                except:
                    duration = 1
                avg_frame_rate = s.get("avg_frame_rate", None)
                width = s.get("width", None)
                height = s.get("height", None)

                if avg_frame_rate is None or width is None or height is None:
                    return None
                fps_split = avg_frame_rate.split("/")

                if len(fps_split) == 2 and int(fps_split[1]) > 0:
                    fps = float(fps_split[0]) / float(fps_split[1])
                elif len(fps_split) == 1:

                    fps = float(fps_split[0])
                return {"fps": fps, "width": width, "height": height, "size": (width, height), "duration": duration}

    except:
        return None


class VideoDecoder:
    # TODO: videos with sample aspect ratio (SAR) not equal to 1:1 are loaded with wrong shape
    def __init__(self, path, max_dimension=None, fps=None):
        """Provides an iterator over the frames of a video.

        Args:
            path (str): Path to the video file.
            max_dimension (Union[List, int], optional): Resize the video by providing either 
                - a list of shape [width, height] or
                - an int that depicts the longer side of the frame.
                Defaults to None.
            fps (int, optional): Frames per second. Defaults to None.
        """
        self._path = path
        self._max_dimension = max_dimension
        self._fps = fps

        self._meta = parse_meta_ffmpeg(path)
        if self._meta is None:
            # self._meta = ffmpeg.probe(path)
            self._meta = iio.immeta(path, plugin="FFMPEG")

        self._size = self._meta.get("size")

        if self._fps is None:
            self._fps = self._meta.get("fps")

        self._duration = self._meta.get("duration")

    def __iter__(self):

        if self._max_dimension is not None:

            if isinstance(self._max_dimension, (list, tuple)):
                res = self._max_dimension
            else:
                res = max(self._size[1], self._size[0])

                scale = min(self._max_dimension / res, 1)
                res = (round(self._size[0] * scale), round(self._size[1] * scale))

            video_reader = iio.imiter(
                self._path,
                plugin="pyav",
                format="rgb24",
                filter_sequence=[
                    ("fps", {"fps": f"{self._fps}", "round": "up"}),
                    ("scale", {"width": f"{res[0]}", "height": f"{res[1]}"}),
                ],
            )

        else:
            video_reader = iio.imiter(
                self._path,
                plugin="pyav",
                format="rgb24",
                filter_sequence=[
                    ("fps", {"fps": f"{self._fps}", "round": "up"}),
                ],
            )

        for i, frame in enumerate(video_reader):
            yield {"time": i / self._fps, "index": i, "frame": frame, "ref_id": self._path}

    def fps(self):
        return self._fps

    def duration(self):
        return self._duration


class VideoBatcher:
    def __init__(self, video_decoder: VideoDecoder, batch_size=8):
        self.video_decoder = video_decoder
        self.batch_size = batch_size

    def __iter__(self):
        cache = []
        for x in self.video_decoder:
            cache.append(x)

            if len(cache) >= self.batch_size:
                yield {
                    "time": [x["time"] for x in cache],
                    "index": [x["index"] for x in cache],
                    "frame": np.stack([x["frame"] for x in cache]),
                    "ref_id": [x["ref_id"] for x in cache],
                }
                cache = []

        if len(cache) >= self.batch_size:
            yield {
                "time": [x["time"] for x in cache],
                "index": [x["index"] for x in cache],
                "frame": np.stack([x["frame"] for x in cache]),
                "ref_id": [x["ref_id"] for x in cache],
            }

    def fps(self):
        return self.video_decoder._fps

    def duration(self):
        return self.video_decoder._duration

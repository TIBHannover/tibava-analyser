import imageio.v3 as iio
import ffmpeg
import code
import traceback
import sys
import av


def parse_meta_av(path, **kwargs):
    try:
        fh = av.open(path)
        stream = fh.streams.video[0]
        frame = next(fh.decode(video=0))
        frame = frame.reformat(format="rgb24")
        # print(dir(fh))
        # print(frame)
        # print(dir(frame))
        # print("########")
        # print(frame.interlaced_frame)
        # print(frame.format)
        # print(frame.planes)
        # print(frame.pict_type)

        # print(frame.width)
        # print(frame.height)
        # print(frame.to_ndarray().shape)
        # iio.imwrite(os.path.join(test_path, "test_out.jpg"), frame.to_ndarray())
        # print(stream.duration)
        # print(stream.time_base)
        # print(fh.size)
        # print(float(stream.duration * stream.time_base))
        # print(stream.average_rate)
        # print(stream.guessed_rate)
        return {
            "fps": stream.average_rate,
            "width": frame.width,
            "height": frame.height,
            "size": (frame.width, frame.height),
            "duration": float(stream.duration * stream.time_base),
        }

    except:
        return None


def parse_meta_imageio(path, **kwargs):
    try:
        # TODO
        meta = iio.immeta(path, plugin="FFMPEG", **kwargs)
        return {
            "fps": meta.average_rate,
            "width": meta.width,
            "height": meta.height,
            "size": (meta.width, meta.height),
            "duration": float(meta.duration * meta.time_base),
        }

    except:
        return None


# def parse_meta_ffmpeg(path):
#     try:
#         meta = ffmpeg.probe(path)

#         streams = meta.get("streams", [])
#         for s in streams:
#             if s.get("codec_type") == "video":

#                 duration = s.get("duration", None)
#                 try:
#                     duration = float(duration)
#                 except:
#                     duration = 1
#                 avg_frame_rate = s.get("avg_frame_rate", None)
#                 width = s.get("width", None)
#                 height = s.get("height", None)

#                 if avg_frame_rate is None or width is None or height is None:
#                     return None
#                 fps_split = avg_frame_rate.split("/")

#                 if len(fps_split) == 2 and int(fps_split[1]) > 0:
#                     fps = float(fps_split[0]) / float(fps_split[1])
#                 elif len(fps_split) == 1:

#                     fps = float(fps_split[0])
#                 return {"fps": fps, "width": width, "height": height, "size": (width, height), "duration": duration}

#     except:
#         return None

import pdb


class VideoDecoder:
    # TODO: videos with sample aspect ratio (SAR) not equal to 1:1 are loaded with wrong shape
    def __init__(self, path, max_dimension=None, fps=None, **kwargs):
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

        self._meta = parse_meta_av(path)

        self._size = self._meta.get("size")

        self._real_fps = self._meta.get("fps")
        self._duration = self._meta.get("duration")

        self._kwargs = kwargs

    def __iter__(self):
        filter_sequence = []

        if self._fps is not None:
            filter_sequence.append(("fps", {"fps": f"{self._fps}", "round": "up"}))

        if self._max_dimension is not None:

            if isinstance(self._max_dimension, (list, tuple)):
                res = self._max_dimension
            else:
                res = max(self._size[1], self._size[0])

                scale = min(self._max_dimension / res, 1)
                res = (round(self._size[0] * scale), round(self._size[1] * scale))
            filter_sequence.append(("scale", {"width": f"{res[0]}", "height": f"{res[1]}"}))
        video_reader = iio.imiter(
            self._path,
            plugin="pyav",
            format="rgb24",
            filter_sequence=filter_sequence,
            **self._kwargs,
        )
        fps = self._real_fps if self._fps is None else self._fps
        for i, frame in enumerate(video_reader):
            yield {"time": i / fps, "index": i, "frame": frame, "ref_id": self._path}

    def fps(self):
        return float(self._real_fps if self._fps is None else self._fps)

    def duration(self):
        return self._duration

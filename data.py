import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Type, Iterator

import msgpack
import numpy.typing as npt
import numpy as np

from analyser import analyser_pb2


@dataclass
class PluginData:
    id: str = dataclass.field(default_factory=lambda: uuid.uuid4().hex)

    def dump(self):
        pass

    def load(self):
        pass


@dataclass
class VideoData(PluginData):
    path: str = None
    ext: str = None


@dataclass
class ImageData(PluginData):
    path: str = None
    time: float = None
    ext: str = None


@dataclass
class Shot:
    start: float
    end: float


@dataclass
class ShotsData(PluginData):
    shots: List[Shot] = dataclass.field(default_factory=list)


@dataclass
class AudioData(PluginData):
    path: str = None
    ext: str = None


@dataclass
class ScalarData(PluginData):
    x: npt.NDArray = dataclass.field(default_factory=np.ndarray)
    time: List[float] = dataclass.field(default_factory=list)


@dataclass
class HistData(PluginData):
    x: npt.NDArray = dataclass.field(default_factory=np.ndarray)
    time: List[float] = dataclass.field(default_factory=list)


def data_from_proto_stream(proto, data_dir=None):
    if proto.type == analyser_pb2.VIDEO_DATA:
        data = VideoData()
        if hasattr(proto, "ext"):
            data.ext = proto.ext
        if data_dir:
            data.path = os.path.join(data_dir)

        return data


def data_from_proto(proto, data_dir=None):
    if proto.type == analyser_pb2.VIDEO_DATA:
        data = VideoData()
        if hasattr(proto, "ext"):
            data.ext = proto.ext
        if data_dir:
            data.path = os.path.join(data_dir)

        return data


class DataManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_from_stream(self, data: Iterator[Any]):

        datastream = iter(data)
        firstpkg = next(datastream)
        if firstpkg.type == analyser_pb2.VIDEO_DATA:
            data = VideoData()
            if hasattr(firstpkg, "ext"):
                data.ext = firstpkg.ext
            else:
                data.ext = "mp4"
            if self.data_dir:
                data.path = os.path.join(self.data_dir, f"{data.id}.{data.ext}")

            with open(data.path, "wb") as f:
                f.write(firstpkg.data_encoded)  # write first package
                for x in datastream:
                    f.write(x.data_encoded)

            return data

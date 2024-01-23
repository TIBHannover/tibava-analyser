import logging
from typing import List
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

from ..manager import DataManager
from ..data import Data
from .face_data import FacesData, FaceData
from .keypoint_data import KpssData, KpsData
from .bounding_box_data import BboxesData, BboxData
from .image_data import ImagesData, ImageData
from analyser.proto import analyser_pb2
from .image_embedding import ImageEmbedding


@dataclass(kw_only=True)
class Cluster(Data):
    object_refs: List[str] = field(default_factory=list)
    sample_object_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        meta = super().to_dict()
        return {
            **meta,
            "object_refs": self.object_refs,
            "sample_object_refs": self.sample_object_refs,
        }


@DataManager.export("ClusterData", analyser_pb2.CLUSTER_DATA)
@dataclass(kw_only=True)
class ClusterData(Data):
    type: str = field(default="ClusterData")
    clusters: List[Cluster] = field(default_factory=list)

    def load(self) -> None:
        super().load()
        assert self.check_fs(), "No filesystem handler installed"

        data = self.load_dict("cluster_data.yml")
        self.clusters = [Cluster(**x) for x in data.get("cluster")]

    def save(self) -> None:
        super().save()
        assert self.check_fs(), "No filesystem handler installed"
        assert self.fs.mode == "w", "Data packet is open read only"

        self.save_dict(
            "cluster_data.yml",
            {
                "cluster": [c.to_dict() for c in self.clusters],
            },
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "cluster": [c.to_dict() for c in self.clusters],
        }

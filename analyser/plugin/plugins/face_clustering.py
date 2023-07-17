from analyser.plugin.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import ScalarData, ImageEmbeddings, FaceClusterData

import logging
import numpy as np
from scipy.spatial.distance import cdist
from analyser.data import DataManager, Data
from scipy.cluster.hierarchy import fclusterdata

from typing import Callable, Optional, Dict


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"min_threshold": None, "max_threshold": None}

requires = {
    "embeddings": ImageEmbeddings,
}

provides = {
    "face_cluster_data": FaceClusterData,
}


@AnalyserPluginManager.export("face_clustering")
class FaceClustering(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        with inputs["embeddings"] as face_embeddings, inputs["faces"] as faces, data_manager.create_data("FaceClusterData") as output_data:

            embeddings = [qf.embedding for qf in face_embeddings.embeddings]
            faces = [f.id for f in faces.faces]

            cluster_threshold=0.4
            metric="cosine"
            result = fclusterdata(X=embeddings, t=cluster_threshold, criterion="distance", metric=metric)
            # result format: list of cluster ids [1 2 1 3]
            # sort face refs into clusters

            output_data.clusters = [[] for _ in np.unique(result)]
            for id, cluster_id in enumerate(result):
                output_data.clusters[cluster_id-1].append(faces[id])
         
            return {"face_cluster_data": output_data}

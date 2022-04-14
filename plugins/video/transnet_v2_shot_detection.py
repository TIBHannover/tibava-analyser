
from analyser.plugins import VideoPlugin, VideoPluginManager

@VideoPluginManager.export("TransnetV2ShotDetection")
class TransnetV2ShotDetection(VideoPlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "byol_wikipedia",
        "model_device": "gpu",
        "model_file": "/home/matthias/transnetv2.mar"
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(TransnetV2ShotDetection, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.max_tries = self.config["max_tries"]

        try_count = self.max_tries
        while try_count > 0:
            try:
                self.con = rai.Client(host=self.host, port=self.port)

                if not self.check_rai():
                    self.register_rai()
                return
            except:
                try_count -= 1
                time.sleep(4)

    def register_rai(self):
        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="torch",
            device=self.model_device,
            data=model,
            batch=16,
        )

    def check_rai(self):
        result = self.con.modelscan()
        if self.model_name in [x[0] for x in result]:
            return True
        return False

    def call(self, entries):

        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            # image = image_from_proto(entry)
            image = entry
            image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(self.model_name, f"image_{job_id}", f"output_{job_id}")
            output = self.con.tensorget(f"output_{job_id}")[0, ...]
            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"output_{job_id}")

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="byol_embedding", binary=output_bin_str, feature=output.tolist()
                    ),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)

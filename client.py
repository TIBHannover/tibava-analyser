import os
import sys
import re
import argparse
import logging
import mimetypes
from typing import Iterator, Any

import grpc
import json
from analyser import analyser_pb2, analyser_pb2_grpc

from google.protobuf.json_format import MessageToJson

from analyser.data import DataManager

import time
import msgpack


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=50051, type=int)
    parser.add_argument(
        "-t", "--task", choices=["list_plugins", "upload_file", "run_plugin", "download_data", "get_plugin_status"]
    )
    parser.add_argument("--path")
    parser.add_argument("--plugin")
    parser.add_argument("--inputs")
    parser.add_argument("--parameters")
    parser.add_argument("--id")
    args = parser.parse_args()
    return args


class AnalyserClient:
    def __init__(self, host, port, manager=None):
        self.host = host
        self.port = port
        self.manager = manager
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")

    def list_plugins(self):
        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        response = stub.list_plugins(analyser_pb2.ListPluginsRequest())
        result = {}

        # for plugin in response.plugins:
        #     print(MessageToJson(plugin))

        return result

    def upload_data(self, data):
        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        def generate_requests(data, chunk_size=128 * 1024):
            if self.manager is None:
                data_manager = DataManager()
            else:
                data_manager = self.manager
            data_manager.save(data)
            data = data_manager.load(data.id)
            """Lazy function (generator) to read a file piece by piece.
            Default chunk size: 1k"""
            for x in data_manager.dump_to_stream(data):
                yield analyser_pb2.UploadDataRequest(type=x["type"], data_encoded=x["data_encoded"])

        response = stub.upload_data(generate_requests(data))

        if response.success:
            return response.id

        logging.error("Error while copying data ...")
        return None

    def upload_file(self, path):
        mimetype = mimetypes.guess_type(path)
        if re.match(r"video/*", mimetype[0]):
            data_type = analyser_pb2.VIDEO_DATA
        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        def generate_requests(file_object, chunk_size=128 * 1024):
            """Lazy function (generator) to read a file piece by piece.
            Default chunk size: 1k"""
            with open(file_object, "rb") as bytestream:
                while True:
                    data = bytestream.read(chunk_size)
                    if not data:
                        break
                    yield analyser_pb2.UploadDataRequest(type=data_type, data_encoded=data)

        response = stub.upload_data(generate_requests(path))

        if response.success:
            return response.id

        logging.error("Error while copying data ...")
        return None

    def run_plugin(self, plugin, inputs, parameters):

        run_request = analyser_pb2.RunPluginRequest()
        run_request.plugin = plugin
        for i in inputs:
            x = run_request.inputs.add()
            x.name = i.get("name")
            x.id = i.get("id")

        for i in parameters:
            x = run_request.parameters.add()
            x.name = i.get("name")
            x.value = str(i.get("value"))
            if isinstance(i.get("value"), float):
                x.type = analyser_pb2.FLOAT_TYPE
            if isinstance(i.get("value"), int):
                x.type = analyser_pb2.INT_TYPE
            if isinstance(i.get("value"), str):
                x.type = analyser_pb2.STRING_TYPE

        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        response = stub.run_plugin(run_request)

        if response.success:
            return response.id

        logging.error("Error while run plugin ...")
        return None

    def get_plugin_status(self, job_id):

        get_plugin_request = analyser_pb2.GetPluginStatusRequest(id=job_id)

        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        response = stub.get_plugin_status(get_plugin_request)

        return response

    def get_plugin_results(self, job_id, timeout=None):
        result = None
        start_time = time.time()
        while True:
            if timeout:
                print(time.time() - start_time)
                if time.time() - start_time > timeout:
                    return None
            result = self.get_plugin_status(job_id)

            if result.status == analyser_pb2.GetPluginStatusResponse.WAITING:
                time.sleep(0.5)
                continue
            if result.status == analyser_pb2.GetPluginStatusResponse.RUNNING:
                time.sleep(0.5)
                continue
            elif result.status == analyser_pb2.GetPluginStatusResponse.ERROR:
                logging.error("Job is crashing")
                return
            elif result.status == analyser_pb2.GetPluginStatusResponse.DONE:
                break

        return result

    def download_data(self, data_id, output_path):

        download_data_request = analyser_pb2.DownloadDataRequest(id=data_id)

        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        response = stub.download_data(download_data_request)
        data = DataManager(output_path).load_from_stream(response)
        return data

    def download_data_to_blob(self, data_id, output_path):

        download_data_request = analyser_pb2.DownloadDataRequest(id=data_id)

        
        stub = analyser_pb2_grpc.AnalyserStub(self.channel)

        response = stub.download_data(download_data_request)
        with open(os.path.join(output_path, f"{data_id}.bin"), "wb") as f:
            for x in response:
                f.write(msgpack.packb({"d": x.SerializeToString()}))

        return os.path.join(output_path, f"{data_id}.bin")


def main():
    args = parse_args()

    level = logging.ERROR
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)
    client = AnalyserClient(args.host, args.port)

    if args.task == "list_plugins":
        result = client.list_plugins()
        print(result)

    if args.task == "upload_file":
        result = client.upload_file(args.path)
        print(result)

    if args.task == "run_plugin":
        result = client.run_plugin(args.plugin, json.loads(args.inputs), json.loads(args.parameters))
        print(result)

    if args.task == "get_plugin_status":
        result = client.get_plugin_status(args.id)
        print(result)

    if args.task == "download_data":
        result = client.download_data(args.id, args.path)
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

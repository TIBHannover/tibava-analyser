from __future__ import annotations

import os
import logging
import json
import tempfile
import hashlib
from typing import Any, Iterator, List
from collections.abc import Iterable

from dataclasses import field

from .data import Data
from .fs_handler import ZipFSHandler
from .utils import create_data_path, generate_id


class DataManager:
    _data_name_lut = {}
    _data_enum_lut = {}
    _data_minetype_lut = {}

    def __init__(self, data_dir=None, cache=None):
        self.cache = cache
        if not data_dir:
            data_dir = tempfile.mkdtemp()
        self.data_dir = data_dir

    @classmethod
    def export(cls, name: str, enum_value: int, minetype: List[str] = None):
        def export_helper(data):
            cls._data_name_lut[name] = data
            cls._data_enum_lut[enum_value] = data
            if minetype:
                for x in minetype:
                    cls._data_minetype_lut[x] = data
            return data

        return export_helper

    def create_data(self, data_type: str):
        print(self._data_name_lut)
        assert data_type in self._data_name_lut, "Unknown data type {data_type}"

        data = self._data_name_lut[data_type]()
        data_path = create_data_path(self.data_dir, data.id, "zip")
        data._register_fs_handler(ZipFSHandler(data_path, mode="w"))
        return data

    def load(self, data_id: str):
        data_path = create_data_path(self.data_dir, data_id, "zip")

        if not os.path.exists(data_path):
            logging.error("Data not found with data_id {data_id}")
            return None
        data = Data()
        data._register_fs_handler(ZipFSHandler(data_path, mode="r"))
        data_type = None
        with data:
            data_type = data.type

        assert data_type in self._data_name_lut, "Unknown data type {name}"

        data = self._data_name_lut[data_type]()
        data._register_fs_handler(ZipFSHandler(data_path, mode="r"))

        return data

    def load_file_from_stream(self, data_stream: Iterable) -> tuple(Data, str):

        data_stream = iter(data_stream)
        first_pkg = next(data_stream)

        hash_stream = hashlib.sha1()

        if first_pkg.type not in self._data_enum_lut:
            logging.error(f"No data class register with index {first_pkg.type}")
            return None

        data_type = first_pkg.type

        if first_pkg.id is not None and len(first_pkg.id) > 0:
            data_id = first_pkg.id
            if os.path.exists(create_data_path(self.data_dir, data_id, "zip")):
                logging.error(f"Data with id already exists {data_id}")
                return None
            data = self._data_enum_lut[data_type](id=data_id)

        else:
            data = self._data_enum_lut[data_type]()

        assert hasattr(data, "load_file_from_stream"), f"Data {data.type} has no function load_file_from_stream"

        data_path = create_data_path(self.data_dir, data.id, "zip")
        data._register_fs_handler(ZipFSHandler(data_path, mode="w"))

        def data_generator():
            yield first_pkg

            hash_stream.update(first_pkg.data_encoded)
            for x in data_stream:
                hash_stream.update(x.data_encoded)
                yield x

        with data as d:
            d.load_file_from_stream(data_generator())
        print(data)
        print(hash_stream.hexdigest())
        return data, hash_stream.hexdigest()

    def load_data_from_stream(self, data_stream: Iterable) -> tuple(Data, str):

        data_stream = iter(data_stream)
        first_pkg = next(data_stream)

        data_id = first_pkg.id

        hash_stream = hashlib.sha1()

        output_path = create_data_path(self.data_dir, data_id, "zip")

        if os.path.exists(output_path):
            logging.error(f"Data with id already exists {data_id}")
            return None

        def data_generator():
            yield first_pkg.data_encoded

            hash_stream.update(first_pkg.data_encoded)
            for x in data_stream:
                hash_stream.update(x.data_encoded)
                yield x.data_encoded

        with open(output_path, "wb") as f_out:
            for x in data_generator():
                f_out.write(x)

        return self.load(data_id), hash_stream.hexdigest()

    def dump_to_stream(self, data_id: str, chunk_size: int = 131_072) -> Iterator[dict]:
        data_path = create_data_path(self.data_dir, data_id, "zip")

        if not os.path.exists(data_path):
            logging.error("Data not found with id {id}")
            return None

        with open(data_path, "rb") as bytestream:
            while True:
                chunk = bytestream.read(chunk_size)
                if not chunk:
                    break
                yield {"id": data_id, "data_encoded": chunk}

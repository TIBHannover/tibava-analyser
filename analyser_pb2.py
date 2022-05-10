# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: analyser.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x61nalyser.proto\x12\x0ftibava.analyser\"Y\n\x0fPluginParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65\x66\x61ult\x18\x02 \x01(\t\x12\'\n\x04type\x18\x03 \x01(\x0e\x32\x19.tibava.analyser.DataType\"I\n\nPluginData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12-\n\x04type\x18\x02 \x01(\x0e\x32\x1f.tibava.analyser.PluginDataType\"\xbf\x01\n\nPluginInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x34\n\nparameters\x18\x03 \x03(\x0b\x32 .tibava.analyser.PluginParameter\x12-\n\x08requires\x18\x04 \x03(\x0b\x32\x1b.tibava.analyser.PluginData\x12-\n\x08provides\x18\x05 \x03(\x0b\x32\x1b.tibava.analyser.PluginData\"\x14\n\x12ListPluginsRequest\"@\n\x10ListPluginsReply\x12,\n\x07plugins\x18\x01 \x03(\x0b\x32\x1b.tibava.analyser.PluginInfo\"R\n\x0b\x44\x61taRequest\x12\x14\n\x0c\x64\x61ta_encoded\x18\x01 \x01(\x0c\x12-\n\x04type\x18\x02 \x01(\x0e\x32\x1f.tibava.analyser.PluginDataType\"+\n\x0c\x44\x61taResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08*J\n\x08\x44\x61taType\x12\x0f\n\x0bUNKOWN_TYPE\x10\x00\x12\x0f\n\x0bSTRING_TYPE\x10\x01\x12\x0c\n\x08INT_TYPE\x10\x02\x12\x0e\n\nFLOAT_TYPE\x10\x03*O\n\x0ePluginDataType\x12\x0f\n\x0bUNKOWN_DATA\x10\x00\x12\x0e\n\nVIDEO_DATA\x10\x01\x12\x0e\n\nIMAGE_DATA\x10\x02\x12\x0c\n\x08ROI_DATA\x10\x03\x32\xb0\x01\n\x08\x41nalyser\x12J\n\tcopy_data\x12\x1c.tibava.analyser.DataRequest\x1a\x1d.tibava.analyser.DataResponse(\x01\x12X\n\x0clist_plugins\x12#.tibava.analyser.ListPluginsRequest\x1a!.tibava.analyser.ListPluginsReply\"\x00\x42\x02P\x01\x62\x06proto3')

_DATATYPE = DESCRIPTOR.enum_types_by_name['DataType']
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
_PLUGINDATATYPE = DESCRIPTOR.enum_types_by_name['PluginDataType']
PluginDataType = enum_type_wrapper.EnumTypeWrapper(_PLUGINDATATYPE)
UNKOWN_TYPE = 0
STRING_TYPE = 1
INT_TYPE = 2
FLOAT_TYPE = 3
UNKOWN_DATA = 0
VIDEO_DATA = 1
IMAGE_DATA = 2
ROI_DATA = 3


_PLUGINPARAMETER = DESCRIPTOR.message_types_by_name['PluginParameter']
_PLUGINDATA = DESCRIPTOR.message_types_by_name['PluginData']
_PLUGININFO = DESCRIPTOR.message_types_by_name['PluginInfo']
_LISTPLUGINSREQUEST = DESCRIPTOR.message_types_by_name['ListPluginsRequest']
_LISTPLUGINSREPLY = DESCRIPTOR.message_types_by_name['ListPluginsReply']
_DATAREQUEST = DESCRIPTOR.message_types_by_name['DataRequest']
_DATARESPONSE = DESCRIPTOR.message_types_by_name['DataResponse']
PluginParameter = _reflection.GeneratedProtocolMessageType('PluginParameter', (_message.Message,), {
  'DESCRIPTOR' : _PLUGINPARAMETER,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginParameter)
  })
_sym_db.RegisterMessage(PluginParameter)

PluginData = _reflection.GeneratedProtocolMessageType('PluginData', (_message.Message,), {
  'DESCRIPTOR' : _PLUGINDATA,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginData)
  })
_sym_db.RegisterMessage(PluginData)

PluginInfo = _reflection.GeneratedProtocolMessageType('PluginInfo', (_message.Message,), {
  'DESCRIPTOR' : _PLUGININFO,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginInfo)
  })
_sym_db.RegisterMessage(PluginInfo)

ListPluginsRequest = _reflection.GeneratedProtocolMessageType('ListPluginsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTPLUGINSREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.ListPluginsRequest)
  })
_sym_db.RegisterMessage(ListPluginsRequest)

ListPluginsReply = _reflection.GeneratedProtocolMessageType('ListPluginsReply', (_message.Message,), {
  'DESCRIPTOR' : _LISTPLUGINSREPLY,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.ListPluginsReply)
  })
_sym_db.RegisterMessage(ListPluginsReply)

DataRequest = _reflection.GeneratedProtocolMessageType('DataRequest', (_message.Message,), {
  'DESCRIPTOR' : _DATAREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.DataRequest)
  })
_sym_db.RegisterMessage(DataRequest)

DataResponse = _reflection.GeneratedProtocolMessageType('DataResponse', (_message.Message,), {
  'DESCRIPTOR' : _DATARESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.DataResponse)
  })
_sym_db.RegisterMessage(DataResponse)

_ANALYSER = DESCRIPTOR.services_by_name['Analyser']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'P\001'
  _DATATYPE._serialized_start=612
  _DATATYPE._serialized_end=686
  _PLUGINDATATYPE._serialized_start=688
  _PLUGINDATATYPE._serialized_end=767
  _PLUGINPARAMETER._serialized_start=35
  _PLUGINPARAMETER._serialized_end=124
  _PLUGINDATA._serialized_start=126
  _PLUGINDATA._serialized_end=199
  _PLUGININFO._serialized_start=202
  _PLUGININFO._serialized_end=393
  _LISTPLUGINSREQUEST._serialized_start=395
  _LISTPLUGINSREQUEST._serialized_end=415
  _LISTPLUGINSREPLY._serialized_start=417
  _LISTPLUGINSREPLY._serialized_end=481
  _DATAREQUEST._serialized_start=483
  _DATAREQUEST._serialized_end=565
  _DATARESPONSE._serialized_start=567
  _DATARESPONSE._serialized_end=610
  _ANALYSER._serialized_start=770
  _ANALYSER._serialized_end=946
# @@protoc_insertion_point(module_scope)

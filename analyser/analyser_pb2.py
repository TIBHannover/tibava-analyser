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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x61nalyser.proto\x12\x0ftibava.analyser\"]\n\x13PluginInfoParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65\x66\x61ult\x18\x02 \x01(\t\x12\'\n\x04type\x18\x03 \x01(\x0e\x32\x19.tibava.analyser.DataType\"M\n\x0ePluginInfoData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12-\n\x04type\x18\x02 \x01(\x0e\x32\x1f.tibava.analyser.PluginDataType\")\n\rRunPluginData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\"\xcb\x01\n\nPluginInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x38\n\nparameters\x18\x03 \x03(\x0b\x32$.tibava.analyser.PluginInfoParameter\x12\x31\n\x08requires\x18\x04 \x03(\x0b\x32\x1f.tibava.analyser.PluginInfoData\x12\x31\n\x08provides\x18\x05 \x03(\x0b\x32\x1f.tibava.analyser.PluginInfoData\"\x14\n\x12ListPluginsRequest\"@\n\x10ListPluginsReply\x12,\n\x07plugins\x18\x01 \x03(\x0b\x32\x1b.tibava.analyser.PluginInfo\"5\n\x11UploadDataRequest\x12\x14\n\x0c\x64\x61ta_encoded\x18\x01 \x01(\x0c\x12\n\n\x02id\x18\x04 \x01(\t\"?\n\x12UploadDataResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0c\n\x04hash\x18\x03 \x01(\t\"\x83\x01\n\x11UploadFileRequest\x12\x14\n\x0c\x64\x61ta_encoded\x18\x01 \x01(\x0c\x12-\n\x04type\x18\x02 \x01(\x0e\x32\x1f.tibava.analyser.PluginDataType\x12\x0b\n\x03\x65xt\x18\x03 \x01(\t\x12\x10\n\x08\x66ilename\x18\x04 \x01(\t\x12\n\n\x02id\x18\x05 \x01(\t\"?\n\x12UploadFileResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0c\n\x04hash\x18\x03 \x01(\t\"!\n\x13\x44ownloadDataRequest\x12\n\n\x02id\x18\x01 \x01(\t\"F\n\x14\x44ownloadDataResponse\x12\x14\n\x0c\x64\x61ta_encoded\x18\x01 \x01(\x0c\x12\x0c\n\x04hash\x18\x04 \x01(\t\x12\n\n\x02id\x18\x05 \x01(\t\"\x1e\n\x10\x43heckDataRequest\x12\n\n\x02id\x18\x01 \x01(\t\"1\n\x11\x43heckDataResponse\x12\x0e\n\x06\x65xists\x18\x01 \x01(\x08\x12\x0c\n\x04hash\x18\x02 \x01(\t\"W\n\x0fPluginParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\x12\'\n\x04type\x18\x03 \x01(\x0e\x32\x19.tibava.analyser.DataType\"\x88\x01\n\x10RunPluginRequest\x12\x0e\n\x06plugin\x18\x01 \x01(\t\x12.\n\x06inputs\x18\x02 \x03(\x0b\x32\x1e.tibava.analyser.RunPluginData\x12\x34\n\nparameters\x18\x03 \x03(\x0b\x32 .tibava.analyser.PluginParameter\"0\n\x11RunPluginResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\"$\n\x16GetPluginStatusRequest\x12\n\n\x02id\x18\x01 \x01(\t\"\xe3\x01\n\x17GetPluginStatusResponse\x12?\n\x06status\x18\x01 \x01(\x0e\x32/.tibava.analyser.GetPluginStatusResponse.Status\x12/\n\x07outputs\x18\x02 \x03(\x0b\x32\x1e.tibava.analyser.RunPluginData\x12\x10\n\x08progress\x18\x03 \x01(\x02\"D\n\x06Status\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05\x45RROR\x10\x01\x12\x08\n\x04\x44ONE\x10\x02\x12\x0b\n\x07RUNNING\x10\x03\x12\x0b\n\x07WAITING\x10\x04\"-\n\x0fRunPipelineData\x12\x0e\n\x06symbol\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\"1\n\x11PipelineSymbolMap\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06symbol\x18\x02 \x01(\t\"\xbf\x01\n\x0ePipelinePlugin\x12\x0e\n\x06plugin\x18\x01 \x01(\t\x12\x32\n\x06inputs\x18\x02 \x03(\x0b\x32\".tibava.analyser.PipelineSymbolMap\x12\x33\n\x07outputs\x18\x03 \x03(\x0b\x32\".tibava.analyser.PipelineSymbolMap\x12\x34\n\nparameters\x18\x04 \x03(\x0b\x32 .tibava.analyser.PluginParameter\"x\n\x12RunPipelineRequest\x12\x30\n\x06inputs\x18\x01 \x03(\x0b\x32 .tibava.analyser.RunPipelineData\x12\x30\n\x07plugins\x18\x03 \x03(\x0b\x32\x1f.tibava.analyser.PipelinePlugin\"2\n\x13RunPipelineResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\"&\n\x18GetPieplineStatusRequest\x12\n\n\x02id\x18\x01 \x01(\t\"\xe9\x01\n\x19GetPieplineStatusResponse\x12\x41\n\x06status\x18\x01 \x01(\x0e\x32\x31.tibava.analyser.GetPieplineStatusResponse.Status\x12\x31\n\x07outputs\x18\x02 \x03(\x0b\x32 .tibava.analyser.RunPipelineData\x12\x10\n\x08progress\x18\x03 \x01(\x02\"D\n\x06Status\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05\x45RROR\x10\x01\x12\x08\n\x04\x44ONE\x10\x02\x12\x0b\n\x07RUNNING\x10\x03\x12\x0b\n\x07WAITING\x10\x04*Y\n\x08\x44\x61taType\x12\x0f\n\x0bUNKOWN_TYPE\x10\x00\x12\x0f\n\x0bSTRING_TYPE\x10\x01\x12\x0c\n\x08INT_TYPE\x10\x02\x12\x0e\n\nFLOAT_TYPE\x10\x03\x12\r\n\tBOOL_TYPE\x10\x04*\x82\x03\n\x0ePluginDataType\x12\x0f\n\x0bUNKOWN_DATA\x10\x00\x12\x0e\n\nVIDEO_DATA\x10\x01\x12\x0e\n\nIMAGE_DATA\x10\x02\x12\x0f\n\x0b\x42\x42OXES_DATA\x10\x03\x12\x0e\n\nAUDIO_DATA\x10\x04\x12\x0f\n\x0bSCALAR_DATA\x10\x05\x12\x0e\n\nSHOTS_DATA\x10\x06\x12\x0f\n\x0bIMAGES_DATA\x10\x07\x12\r\n\tLIST_DATA\x10\x08\x12\x0c\n\x08RGB_DATA\x10\t\x12\r\n\tHIST_DATA\x10\n\x12\x11\n\rRGB_HIST_DATA\x10\x0b\x12\x13\n\x0f\x41NNOTATION_DATA\x10\x0c\x12\x18\n\x14IMAGE_EMBEDDING_DATA\x10\r\x12\x17\n\x13TEXT_EMBEDDING_DATA\x10\x0e\x12\r\n\tKPSS_DATA\x10\x0f\x12\x0e\n\nFACES_DATA\x10\x10\x12\x12\n\x0e\x43ONTAINER_DATA\x10\x11\x12!\n\x1dVIDEO_TEMPORAL_EMBEDDING_DATA\x10\x12\x12\x0f\n\x0bSTRING_DATA\x10\x13\x32\xd1\x06\n\x08\x41nalyser\x12V\n\x0clist_plugins\x12#.tibava.analyser.ListPluginsRequest\x1a!.tibava.analyser.ListPluginsReply\x12X\n\x0bupload_data\x12\".tibava.analyser.UploadDataRequest\x1a#.tibava.analyser.UploadDataResponse(\x01\x12X\n\x0bupload_file\x12\".tibava.analyser.UploadFileRequest\x1a#.tibava.analyser.UploadFileResponse(\x01\x12^\n\rdownload_data\x12$.tibava.analyser.DownloadDataRequest\x1a%.tibava.analyser.DownloadDataResponse0\x01\x12S\n\ncheck_data\x12!.tibava.analyser.CheckDataRequest\x1a\".tibava.analyser.CheckDataResponse\x12S\n\nrun_plugin\x12!.tibava.analyser.RunPluginRequest\x1a\".tibava.analyser.RunPluginResponse\x12\x66\n\x11get_plugin_status\x12\'.tibava.analyser.GetPluginStatusRequest\x1a(.tibava.analyser.GetPluginStatusResponse\x12Y\n\x0crun_pipeline\x12#.tibava.analyser.RunPipelineRequest\x1a$.tibava.analyser.RunPipelineResponse\x12l\n\x13get_pipeline_status\x12).tibava.analyser.GetPieplineStatusRequest\x1a*.tibava.analyser.GetPieplineStatusResponseB\x02P\x01\x62\x06proto3')

_DATATYPE = DESCRIPTOR.enum_types_by_name['DataType']
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
_PLUGINDATATYPE = DESCRIPTOR.enum_types_by_name['PluginDataType']
PluginDataType = enum_type_wrapper.EnumTypeWrapper(_PLUGINDATATYPE)
UNKOWN_TYPE = 0
STRING_TYPE = 1
INT_TYPE = 2
FLOAT_TYPE = 3
BOOL_TYPE = 4
UNKOWN_DATA = 0
VIDEO_DATA = 1
IMAGE_DATA = 2
BBOXES_DATA = 3
AUDIO_DATA = 4
SCALAR_DATA = 5
SHOTS_DATA = 6
IMAGES_DATA = 7
LIST_DATA = 8
RGB_DATA = 9
HIST_DATA = 10
RGB_HIST_DATA = 11
ANNOTATION_DATA = 12
IMAGE_EMBEDDING_DATA = 13
TEXT_EMBEDDING_DATA = 14
KPSS_DATA = 15
FACES_DATA = 16
CONTAINER_DATA = 17
VIDEO_TEMPORAL_EMBEDDING_DATA = 18
STRING_DATA = 19
FACE_CLUSTER_DATA = 20


_PLUGININFOPARAMETER = DESCRIPTOR.message_types_by_name['PluginInfoParameter']
_PLUGININFODATA = DESCRIPTOR.message_types_by_name['PluginInfoData']
_RUNPLUGINDATA = DESCRIPTOR.message_types_by_name['RunPluginData']
_PLUGININFO = DESCRIPTOR.message_types_by_name['PluginInfo']
_LISTPLUGINSREQUEST = DESCRIPTOR.message_types_by_name['ListPluginsRequest']
_LISTPLUGINSREPLY = DESCRIPTOR.message_types_by_name['ListPluginsReply']
_UPLOADDATAREQUEST = DESCRIPTOR.message_types_by_name['UploadDataRequest']
_UPLOADDATARESPONSE = DESCRIPTOR.message_types_by_name['UploadDataResponse']
_UPLOADFILEREQUEST = DESCRIPTOR.message_types_by_name['UploadFileRequest']
_UPLOADFILERESPONSE = DESCRIPTOR.message_types_by_name['UploadFileResponse']
_DOWNLOADDATAREQUEST = DESCRIPTOR.message_types_by_name['DownloadDataRequest']
_DOWNLOADDATARESPONSE = DESCRIPTOR.message_types_by_name['DownloadDataResponse']
_CHECKDATAREQUEST = DESCRIPTOR.message_types_by_name['CheckDataRequest']
_CHECKDATARESPONSE = DESCRIPTOR.message_types_by_name['CheckDataResponse']
_PLUGINPARAMETER = DESCRIPTOR.message_types_by_name['PluginParameter']
_RUNPLUGINREQUEST = DESCRIPTOR.message_types_by_name['RunPluginRequest']
_RUNPLUGINRESPONSE = DESCRIPTOR.message_types_by_name['RunPluginResponse']
_GETPLUGINSTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetPluginStatusRequest']
_GETPLUGINSTATUSRESPONSE = DESCRIPTOR.message_types_by_name['GetPluginStatusResponse']
_RUNPIPELINEDATA = DESCRIPTOR.message_types_by_name['RunPipelineData']
_PIPELINESYMBOLMAP = DESCRIPTOR.message_types_by_name['PipelineSymbolMap']
_PIPELINEPLUGIN = DESCRIPTOR.message_types_by_name['PipelinePlugin']
_RUNPIPELINEREQUEST = DESCRIPTOR.message_types_by_name['RunPipelineRequest']
_RUNPIPELINERESPONSE = DESCRIPTOR.message_types_by_name['RunPipelineResponse']
_GETPIEPLINESTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetPieplineStatusRequest']
_GETPIEPLINESTATUSRESPONSE = DESCRIPTOR.message_types_by_name['GetPieplineStatusResponse']
_GETPLUGINSTATUSRESPONSE_STATUS = _GETPLUGINSTATUSRESPONSE.enum_types_by_name['Status']
_GETPIEPLINESTATUSRESPONSE_STATUS = _GETPIEPLINESTATUSRESPONSE.enum_types_by_name['Status']
PluginInfoParameter = _reflection.GeneratedProtocolMessageType('PluginInfoParameter', (_message.Message,), {
  'DESCRIPTOR' : _PLUGININFOPARAMETER,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginInfoParameter)
  })
_sym_db.RegisterMessage(PluginInfoParameter)

PluginInfoData = _reflection.GeneratedProtocolMessageType('PluginInfoData', (_message.Message,), {
  'DESCRIPTOR' : _PLUGININFODATA,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginInfoData)
  })
_sym_db.RegisterMessage(PluginInfoData)

RunPluginData = _reflection.GeneratedProtocolMessageType('RunPluginData', (_message.Message,), {
  'DESCRIPTOR' : _RUNPLUGINDATA,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPluginData)
  })
_sym_db.RegisterMessage(RunPluginData)

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

UploadDataRequest = _reflection.GeneratedProtocolMessageType('UploadDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADDATAREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.UploadDataRequest)
  })
_sym_db.RegisterMessage(UploadDataRequest)

UploadDataResponse = _reflection.GeneratedProtocolMessageType('UploadDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADDATARESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.UploadDataResponse)
  })
_sym_db.RegisterMessage(UploadDataResponse)

UploadFileRequest = _reflection.GeneratedProtocolMessageType('UploadFileRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADFILEREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.UploadFileRequest)
  })
_sym_db.RegisterMessage(UploadFileRequest)

UploadFileResponse = _reflection.GeneratedProtocolMessageType('UploadFileResponse', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADFILERESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.UploadFileResponse)
  })
_sym_db.RegisterMessage(UploadFileResponse)

DownloadDataRequest = _reflection.GeneratedProtocolMessageType('DownloadDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADDATAREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.DownloadDataRequest)
  })
_sym_db.RegisterMessage(DownloadDataRequest)

DownloadDataResponse = _reflection.GeneratedProtocolMessageType('DownloadDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADDATARESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.DownloadDataResponse)
  })
_sym_db.RegisterMessage(DownloadDataResponse)

CheckDataRequest = _reflection.GeneratedProtocolMessageType('CheckDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _CHECKDATAREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.CheckDataRequest)
  })
_sym_db.RegisterMessage(CheckDataRequest)

CheckDataResponse = _reflection.GeneratedProtocolMessageType('CheckDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _CHECKDATARESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.CheckDataResponse)
  })
_sym_db.RegisterMessage(CheckDataResponse)

PluginParameter = _reflection.GeneratedProtocolMessageType('PluginParameter', (_message.Message,), {
  'DESCRIPTOR' : _PLUGINPARAMETER,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PluginParameter)
  })
_sym_db.RegisterMessage(PluginParameter)

RunPluginRequest = _reflection.GeneratedProtocolMessageType('RunPluginRequest', (_message.Message,), {
  'DESCRIPTOR' : _RUNPLUGINREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPluginRequest)
  })
_sym_db.RegisterMessage(RunPluginRequest)

RunPluginResponse = _reflection.GeneratedProtocolMessageType('RunPluginResponse', (_message.Message,), {
  'DESCRIPTOR' : _RUNPLUGINRESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPluginResponse)
  })
_sym_db.RegisterMessage(RunPluginResponse)

GetPluginStatusRequest = _reflection.GeneratedProtocolMessageType('GetPluginStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETPLUGINSTATUSREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.GetPluginStatusRequest)
  })
_sym_db.RegisterMessage(GetPluginStatusRequest)

GetPluginStatusResponse = _reflection.GeneratedProtocolMessageType('GetPluginStatusResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETPLUGINSTATUSRESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.GetPluginStatusResponse)
  })
_sym_db.RegisterMessage(GetPluginStatusResponse)

RunPipelineData = _reflection.GeneratedProtocolMessageType('RunPipelineData', (_message.Message,), {
  'DESCRIPTOR' : _RUNPIPELINEDATA,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPipelineData)
  })
_sym_db.RegisterMessage(RunPipelineData)

PipelineSymbolMap = _reflection.GeneratedProtocolMessageType('PipelineSymbolMap', (_message.Message,), {
  'DESCRIPTOR' : _PIPELINESYMBOLMAP,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PipelineSymbolMap)
  })
_sym_db.RegisterMessage(PipelineSymbolMap)

PipelinePlugin = _reflection.GeneratedProtocolMessageType('PipelinePlugin', (_message.Message,), {
  'DESCRIPTOR' : _PIPELINEPLUGIN,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.PipelinePlugin)
  })
_sym_db.RegisterMessage(PipelinePlugin)

RunPipelineRequest = _reflection.GeneratedProtocolMessageType('RunPipelineRequest', (_message.Message,), {
  'DESCRIPTOR' : _RUNPIPELINEREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPipelineRequest)
  })
_sym_db.RegisterMessage(RunPipelineRequest)

RunPipelineResponse = _reflection.GeneratedProtocolMessageType('RunPipelineResponse', (_message.Message,), {
  'DESCRIPTOR' : _RUNPIPELINERESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.RunPipelineResponse)
  })
_sym_db.RegisterMessage(RunPipelineResponse)

GetPieplineStatusRequest = _reflection.GeneratedProtocolMessageType('GetPieplineStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETPIEPLINESTATUSREQUEST,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.GetPieplineStatusRequest)
  })
_sym_db.RegisterMessage(GetPieplineStatusRequest)

GetPieplineStatusResponse = _reflection.GeneratedProtocolMessageType('GetPieplineStatusResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETPIEPLINESTATUSRESPONSE,
  '__module__' : 'analyser_pb2'
  # @@protoc_insertion_point(class_scope:tibava.analyser.GetPieplineStatusResponse)
  })
_sym_db.RegisterMessage(GetPieplineStatusResponse)

_ANALYSER = DESCRIPTOR.services_by_name['Analyser']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'P\001'
  _DATATYPE._serialized_start=2343
  _DATATYPE._serialized_end=2432
  _PLUGINDATATYPE._serialized_start=2435
  _PLUGINDATATYPE._serialized_end=2821
  _PLUGININFOPARAMETER._serialized_start=35
  _PLUGININFOPARAMETER._serialized_end=128
  _PLUGININFODATA._serialized_start=130
  _PLUGININFODATA._serialized_end=207
  _RUNPLUGINDATA._serialized_start=209
  _RUNPLUGINDATA._serialized_end=250
  _PLUGININFO._serialized_start=253
  _PLUGININFO._serialized_end=456
  _LISTPLUGINSREQUEST._serialized_start=458
  _LISTPLUGINSREQUEST._serialized_end=478
  _LISTPLUGINSREPLY._serialized_start=480
  _LISTPLUGINSREPLY._serialized_end=544
  _UPLOADDATAREQUEST._serialized_start=546
  _UPLOADDATAREQUEST._serialized_end=599
  _UPLOADDATARESPONSE._serialized_start=601
  _UPLOADDATARESPONSE._serialized_end=664
  _UPLOADFILEREQUEST._serialized_start=667
  _UPLOADFILEREQUEST._serialized_end=798
  _UPLOADFILERESPONSE._serialized_start=800
  _UPLOADFILERESPONSE._serialized_end=863
  _DOWNLOADDATAREQUEST._serialized_start=865
  _DOWNLOADDATAREQUEST._serialized_end=898
  _DOWNLOADDATARESPONSE._serialized_start=900
  _DOWNLOADDATARESPONSE._serialized_end=970
  _CHECKDATAREQUEST._serialized_start=972
  _CHECKDATAREQUEST._serialized_end=1002
  _CHECKDATARESPONSE._serialized_start=1004
  _CHECKDATARESPONSE._serialized_end=1053
  _PLUGINPARAMETER._serialized_start=1055
  _PLUGINPARAMETER._serialized_end=1142
  _RUNPLUGINREQUEST._serialized_start=1145
  _RUNPLUGINREQUEST._serialized_end=1281
  _RUNPLUGINRESPONSE._serialized_start=1283
  _RUNPLUGINRESPONSE._serialized_end=1331
  _GETPLUGINSTATUSREQUEST._serialized_start=1333
  _GETPLUGINSTATUSREQUEST._serialized_end=1369
  _GETPLUGINSTATUSRESPONSE._serialized_start=1372
  _GETPLUGINSTATUSRESPONSE._serialized_end=1599
  _GETPLUGINSTATUSRESPONSE_STATUS._serialized_start=1531
  _GETPLUGINSTATUSRESPONSE_STATUS._serialized_end=1599
  _RUNPIPELINEDATA._serialized_start=1601
  _RUNPIPELINEDATA._serialized_end=1646
  _PIPELINESYMBOLMAP._serialized_start=1648
  _PIPELINESYMBOLMAP._serialized_end=1697
  _PIPELINEPLUGIN._serialized_start=1700
  _PIPELINEPLUGIN._serialized_end=1891
  _RUNPIPELINEREQUEST._serialized_start=1893
  _RUNPIPELINEREQUEST._serialized_end=2013
  _RUNPIPELINERESPONSE._serialized_start=2015
  _RUNPIPELINERESPONSE._serialized_end=2065
  _GETPIEPLINESTATUSREQUEST._serialized_start=2067
  _GETPIEPLINESTATUSREQUEST._serialized_end=2105
  _GETPIEPLINESTATUSRESPONSE._serialized_start=2108
  _GETPIEPLINESTATUSRESPONSE._serialized_end=2341
  _GETPIEPLINESTATUSRESPONSE_STATUS._serialized_start=1531
  _GETPIEPLINESTATUSRESPONSE_STATUS._serialized_end=1599
  _ANALYSER._serialized_start=2824
  _ANALYSER._serialized_end=3673
# @@protoc_insertion_point(module_scope)

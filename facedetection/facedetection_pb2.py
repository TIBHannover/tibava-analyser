# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: facedetection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='facedetection.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13\x66\x61\x63\x65\x64\x65tection.proto\"\x1f\n\x0b\x46\x61\x63\x65Request\x12\x10\n\x08video_id\x18\x01 \x01(\t\"L\n\x0c\x46\x61\x63\x65Response\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x14\n\x05\x66\x61\x63\x65s\x18\x02 \x03(\x0b\x32\x05.Face\x12\x15\n\rmax_num_faces\x18\x03 \x01(\x05\"l\n\x04\x46\x61\x63\x65\x12\x11\n\tframe_idx\x18\x01 \x01(\x05\x12\x0e\n\x06\x62\x62ox_x\x18\x02 \x01(\x05\x12\x0e\n\x06\x62\x62ox_y\x18\x03 \x01(\x05\x12\x0e\n\x06\x62\x62ox_w\x18\x04 \x01(\x05\x12\x0e\n\x06\x62\x62ox_h\x18\x05 \x01(\x05\x12\x11\n\tbbox_area\x18\x06 \x01(\x02\x32;\n\x0c\x46\x61\x63\x65\x44\x65tector\x12+\n\x0c\x64\x65tect_faces\x12\x0c.FaceRequest\x1a\r.FaceResponseb\x06proto3'
)




_FACEREQUEST = _descriptor.Descriptor(
  name='FaceRequest',
  full_name='FaceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_id', full_name='FaceRequest.video_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=54,
)


_FACERESPONSE = _descriptor.Descriptor(
  name='FaceResponse',
  full_name='FaceResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='FaceResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='faces', full_name='FaceResponse.faces', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_num_faces', full_name='FaceResponse.max_num_faces', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=56,
  serialized_end=132,
)


_FACE = _descriptor.Descriptor(
  name='Face',
  full_name='Face',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_idx', full_name='Face.frame_idx', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox_x', full_name='Face.bbox_x', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox_y', full_name='Face.bbox_y', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox_w', full_name='Face.bbox_w', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox_h', full_name='Face.bbox_h', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox_area', full_name='Face.bbox_area', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=242,
)

_FACERESPONSE.fields_by_name['faces'].message_type = _FACE
DESCRIPTOR.message_types_by_name['FaceRequest'] = _FACEREQUEST
DESCRIPTOR.message_types_by_name['FaceResponse'] = _FACERESPONSE
DESCRIPTOR.message_types_by_name['Face'] = _FACE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FaceRequest = _reflection.GeneratedProtocolMessageType('FaceRequest', (_message.Message,), {
  'DESCRIPTOR' : _FACEREQUEST,
  '__module__' : 'facedetection_pb2'
  # @@protoc_insertion_point(class_scope:FaceRequest)
  })
_sym_db.RegisterMessage(FaceRequest)

FaceResponse = _reflection.GeneratedProtocolMessageType('FaceResponse', (_message.Message,), {
  'DESCRIPTOR' : _FACERESPONSE,
  '__module__' : 'facedetection_pb2'
  # @@protoc_insertion_point(class_scope:FaceResponse)
  })
_sym_db.RegisterMessage(FaceResponse)

Face = _reflection.GeneratedProtocolMessageType('Face', (_message.Message,), {
  'DESCRIPTOR' : _FACE,
  '__module__' : 'facedetection_pb2'
  # @@protoc_insertion_point(class_scope:Face)
  })
_sym_db.RegisterMessage(Face)



_FACEDETECTOR = _descriptor.ServiceDescriptor(
  name='FaceDetector',
  full_name='FaceDetector',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=244,
  serialized_end=303,
  methods=[
  _descriptor.MethodDescriptor(
    name='detect_faces',
    full_name='FaceDetector.detect_faces',
    index=0,
    containing_service=None,
    input_type=_FACEREQUEST,
    output_type=_FACERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FACEDETECTOR)

DESCRIPTOR.services_by_name['FaceDetector'] = _FACEDETECTOR

# @@protoc_insertion_point(module_scope)

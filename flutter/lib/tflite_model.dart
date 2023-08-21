// ignore_for_file: non_constant_identifier_names
import 'dart:io';

import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:path_provider/path_provider.dart';
part 'tflite_model.freezed.dart';
part 'tflite_model.g.dart';

// Always change together with Python `train.models.TFLiteModel`.
@freezed
class TFLiteModel with _$TFLiteModel {
  const factory TFLiteModel({
    required int id,
    required String name,
    required String file_path,
    required String? mlmodel_path,
    required List<int> layers_sizes,
  }) = _TFLiteModel;

  factory TFLiteModel.fromJson(Map<String, dynamic> json) =>
      _$TFLiteModelFromJson(json);
}

Future<(String, String)> getModelDir(TFLiteModel model) async {
  String fileUrl;
  if (Platform.isIOS) {
    fileUrl = model.mlmodel_path!;
  } else {
    fileUrl = model.file_path;
  }
  final base = await getApplicationDocumentsDirectory();
  final fileName = fileUrl.split('/').last;
  final fileDir = '${base.path}/models/${model.name}/$fileName';
  return (fileUrl, fileDir);
}

// ignore_for_file: non_constant_identifier_names
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:dart_mappable/dart_mappable.dart';

part 'ml_model.mapper.dart';

// Always change together with Django `train.models.MLModel`.
/// TFLite or Core ML model.
@MappableClass()
class MLModel with MLModelMappable {
  final int id;
  final String name;
  final String? tflite_path;
  final String? coreml_path;
  final List<int>? tflite_layers;
  final List<Map<String, dynamic>>? coreml_layers;
  final bool tflite;
  final bool coreml;

  MLModel({
    required this.id,
    required this.name,
    this.tflite_path,
    this.coreml_path,
    this.tflite_layers,
    this.coreml_layers,
    this.tflite = true,
    this.coreml = false,
  });

  /// Return `(fileUrl, fileDir)`.
  Future<(String, String)> get urlAndDir async {
    final fileUrl = Platform.isIOS ? coreml_path! : tflite_path!;
    final base = await getApplicationDocumentsDirectory();
    final fileName = fileUrl.split('/').last;
    final fileDir = '${base.path}/models/$name/$fileName';
    return (fileUrl, fileDir);
  }

  get nLayers => Platform.isIOS ? coreml_layers!.length : tflite_layers!.length;
}

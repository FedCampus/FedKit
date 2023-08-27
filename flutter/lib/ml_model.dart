// ignore_for_file: non_constant_identifier_names
import 'package:path_provider/path_provider.dart';
import 'package:dart_mappable/dart_mappable.dart';

part 'ml_model.mapper.dart';

abstract class MlModel {
  final int id;
  final String name;
  final String file_path;

  MlModel({
    required this.id,
    required this.name,
    required this.file_path,
  });

  /// Return `(fileUrl, fileDir)`.
  Future<(String, String)> get urlAndDir async {
    final fileUrl = file_path;
    final base = await getApplicationDocumentsDirectory();
    final fileName = fileUrl.split('/').last;
    final fileDir = '${base.path}/models/$name/$fileName';
    return (fileUrl, fileDir);
  }
}

// Always change together with Django `train.models.TFLiteModel`.
@MappableClass()
class TFLiteModel extends MlModel with TFLiteModelMappable {
  final List<int> layers_sizes;

  TFLiteModel({
    required super.id,
    required super.name,
    required super.file_path,
    required this.layers_sizes,
  });
}

// Always change together with Django `train.models.CoreMLModel`.
@MappableClass()
class CoreMLModel extends MlModel with CoreMLModelMappable {
  final List<String> layers_names;

  CoreMLModel({
    required super.id,
    required super.name,
    required super.file_path,
    required this.layers_names,
  });
}

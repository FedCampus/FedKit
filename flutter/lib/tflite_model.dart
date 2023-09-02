// ignore_for_file: non_constant_identifier_names
import 'package:path_provider/path_provider.dart';
import 'package:dart_mappable/dart_mappable.dart';

part 'tflite_model.mapper.dart';

// Always change together with Django `train.models.TFLiteModel`.
/// Also for Core ML model.
///
/// `T` should be `int` on for TFLite models (on Android) and
/// `Map<String, dynamic>` for Core ML models (on iOS).
@MappableClass()
class TFLiteModel<T> with TFLiteModelMappable<T> {
  final int id;
  final String name;
  final String file_path;
  final List<T> layers_sizes;
  final bool is_coreml;

  TFLiteModel({
    required this.id,
    required this.name,
    required this.file_path,
    required this.layers_sizes,
    required this.is_coreml,
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

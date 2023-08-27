// ignore_for_file: non_constant_identifier_names
import 'package:path_provider/path_provider.dart';
import 'package:dart_mappable/dart_mappable.dart';

part 'tflite_model.mapper.dart';

@MappableClass()
class TFLiteModel with TFLiteModelMappable {
  final int id;
  final String name;
  final String file_path;
  final List<int> layers_sizes;

  TFLiteModel({
    required this.id,
    required this.name,
    required this.file_path,
    required this.layers_sizes,
  });
}

Future<(String, String)> getModelDir(TFLiteModel model) async {
  final fileUrl = model.file_path;
  final base = await getApplicationDocumentsDirectory();
  final fileName = fileUrl.split('/').last;
  final fileDir = '${base.path}/models/${model.name}/$fileName';
  return (fileUrl, fileDir);
}

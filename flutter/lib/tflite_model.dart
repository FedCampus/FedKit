// ignore_for_file: non_constant_identifier_names
import 'package:json_annotation/json_annotation.dart';
part 'tflite_model.g.dart';

@JsonSerializable()
class TFLiteModel {
  final int id;
  final String name;
  final String file_path;
  final List<int> layers_sizes;
  TFLiteModel(this.id, this.name, this.file_path, this.layers_sizes);

  factory TFLiteModel.fromJson(Map<String, dynamic> json) =>
      _$TFLiteModelFromJson(json);
  Map<String, dynamic> toJson() => _$TFLiteModelToJson(this);
}

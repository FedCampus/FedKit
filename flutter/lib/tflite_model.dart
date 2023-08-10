// ignore_for_file: non_constant_identifier_names
import 'package:freezed_annotation/freezed_annotation.dart';
part 'tflite_model.freezed.dart';
part 'tflite_model.g.dart';

// Always change together with Python `train.models.TFLiteModel`.
@freezed
class TFLiteModel with _$TFLiteModel {
  const factory TFLiteModel({
    required int id,
    required String name,
    required String file_path,
    required List<int> layers_sizes,
  }) = _TFLiteModel;

  factory TFLiteModel.fromJson(Map<String, dynamic> json) =>
      _$TFLiteModelFromJson(json);
}

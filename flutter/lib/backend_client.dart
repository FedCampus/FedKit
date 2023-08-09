// ignore_for_file: non_constant_identifier_names
import 'package:dio/dio.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:json_annotation/json_annotation.dart';
part 'backend_client.g.dart';

final dio = Dio();

class BackendClient {
  final String url;
  BackendClient(this.url);

  Future<TFLiteModel> advertisedModel(PostAdvertisedData body) async {
    Response res = await dio.post('$url/advertised_model', data: body.toJson());
    return TFLiteModel.fromJson(res.data);
  }
}

@JsonSerializable()
class PostAdvertisedData {
  String data_type;
  PostAdvertisedData(this.data_type);

  factory PostAdvertisedData.fromJson(Map<String, dynamic> json) =>
      _$PostAdvertisedDataFromJson(json);
  Map<String, dynamic> toJson() => _$PostAdvertisedDataToJson(this);
}

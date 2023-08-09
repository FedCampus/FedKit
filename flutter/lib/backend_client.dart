// ignore_for_file: non_constant_identifier_names
import 'package:dio/dio.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:json_annotation/json_annotation.dart';
part 'backend_client.g.dart';

final dio = Dio();

/// All methods could fail.
class BackendClient {
  final String url;
  BackendClient(this.url);

  Future<TFLiteModel> advertisedModel(PostAdvertisedData body) async {
    Response response =
        await dio.post('$url/train/advertised', data: body.toJson());
    return TFLiteModel.fromJson(response.data);
  }

  Future<void> downloadFile(String urlPath, String destination) async {
    Response response = await dio.download('$url/urlPath', destination);
    int statusCode = response.statusCode!;
    if (statusCode < 200 || statusCode >= 300) {
      throw Exception(
          'downloadFile $urlPath -> $destination failed: $response.');
    }
  }

  Future<ServerData> postServer(TFLiteModel model, bool startFresh) async {
    PostServerData body = PostServerData(model.id, startFresh);
    Response response =
        await dio.post('$url/train/server', data: body.toJson());
    return ServerData.fromJson(response.data);
  }

  Future<void> fitInsTelemetry(FitInsTelemetryData body) async {
    await dio.post('$url/telemetry/fit_ins', data: body.toJson());
  }

  Future<void> evaluateInsTelemetry(EvaluateInsTelemetryData body) async {
    await dio.post('$url/telemetry/evaluate_ins', data: body.toJson());
  }
}

@JsonSerializable()
class PostAdvertisedData {
  final String data_type;

  PostAdvertisedData(this.data_type);

  factory PostAdvertisedData.fromJson(Map<String, dynamic> json) =>
      _$PostAdvertisedDataFromJson(json);
  Map<String, dynamic> toJson() => _$PostAdvertisedDataToJson(this);
}

// Always change together with Python `train.data.ServerData`.
@JsonSerializable()
class ServerData {
  final String status;
  final int? session_id;
  final int? port;

  ServerData(this.status, this.session_id, this.port);

  factory ServerData.fromJson(Map<String, dynamic> json) =>
      _$ServerDataFromJson(json);
  Map<String, dynamic> toJson() => _$ServerDataToJson(this);
}

@JsonSerializable()
class PostServerData {
  final int id;
  final bool start_fresh;

  PostServerData(this.id, this.start_fresh);

  factory PostServerData.fromJson(Map<String, dynamic> json) =>
      _$PostServerDataFromJson(json);
  Map<String, dynamic> toJson() => _$PostServerDataToJson(this);
}

// Always change together with Python `telemetry.models.FitInsTelemetryData`.
@JsonSerializable()
class FitInsTelemetryData {
  final int device_id;
  final int session_id;
  final int start;
  final int end;

  FitInsTelemetryData(this.device_id, this.session_id, this.start, this.end);

  factory FitInsTelemetryData.fromJson(Map<String, dynamic> json) =>
      _$FitInsTelemetryDataFromJson(json);
  Map<String, dynamic> toJson() => _$FitInsTelemetryDataToJson(this);
}

// Always change together with Python `telemetry.models.EvaluateInsTelemetryData`.
@JsonSerializable()
class EvaluateInsTelemetryData {
  final int device_id;
  final int session_id;
  final int start;
  final int end;
  final double loss;
  final double accuracy;
  final int test_size;

  EvaluateInsTelemetryData(this.device_id, this.session_id, this.start,
      this.end, this.loss, this.accuracy, this.test_size);

  factory EvaluateInsTelemetryData.fromJson(Map<String, dynamic> json) =>
      _$EvaluateInsTelemetryDataFromJson(json);
  Map<String, dynamic> toJson() => _$EvaluateInsTelemetryDataToJson(this);
}

// ignore_for_file: non_constant_identifier_names
import 'package:dio/dio.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
part 'backend_client.freezed.dart';
part 'backend_client.g.dart';

final dio = Dio();

/// All methods could fail.
class BackendClient {
  final String url;
  const BackendClient(this.url);

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
    PostServerData body = PostServerData(id: model.id, start_fresh: startFresh);
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

// Add other imports if needed

@freezed
class PostAdvertisedData with _$PostAdvertisedData {
  const factory PostAdvertisedData({
    required String data_type,
  }) = _PostAdvertisedData;

  factory PostAdvertisedData.fromJson(Map<String, dynamic> json) =>
      _$PostAdvertisedDataFromJson(json);
}

@freezed
class ServerData with _$ServerData {
  const factory ServerData({
    required String status,
    int? session_id,
    int? port,
  }) = _ServerData;

  factory ServerData.fromJson(Map<String, dynamic> json) =>
      _$ServerDataFromJson(json);
}

@freezed
class PostServerData with _$PostServerData {
  const factory PostServerData({
    required int id,
    required bool start_fresh,
  }) = _PostServerData;

  factory PostServerData.fromJson(Map<String, dynamic> json) =>
      _$PostServerDataFromJson(json);
}

@freezed
class FitInsTelemetryData with _$FitInsTelemetryData {
  const factory FitInsTelemetryData({
    required int device_id,
    required int session_id,
    required int start,
    required int end,
  }) = _FitInsTelemetryData;

  factory FitInsTelemetryData.fromJson(Map<String, dynamic> json) =>
      _$FitInsTelemetryDataFromJson(json);
}

@freezed
class EvaluateInsTelemetryData with _$EvaluateInsTelemetryData {
  const factory EvaluateInsTelemetryData({
    required int device_id,
    required int session_id,
    required int start,
    required int end,
    required double loss,
    required double accuracy,
    required int test_size,
  }) = _EvaluateInsTelemetryData;

  factory EvaluateInsTelemetryData.fromJson(Map<String, dynamic> json) =>
      _$EvaluateInsTelemetryDataFromJson(json);
}

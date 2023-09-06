// ignore_for_file: non_constant_identifier_names
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:fed_kit/ml_model.dart';
import 'package:dart_mappable/dart_mappable.dart';
part 'backend_client.mapper.dart';

final dio = Dio();

/// All methods could fail.
class BackendClient {
  final String url;
  const BackendClient(this.url);

  Future<MLModel> whichModel(PostAdvertisedData body) async {
    Response response =
        await dio.post('$url/train/advertised', data: body.toJson());
    if (Platform.isIOS) {
      return MLModelMapper.fromMap(response.data);
    } else {
      return MLModelMapper.fromMap(response.data);
    }
  }

  Future<void> downloadFile(String urlPath, String destination) async {
    Response response = await dio.download('$url/$urlPath', destination);
    int statusCode = response.statusCode!;
    if (statusCode < 200 || statusCode >= 300) {
      throw Exception(
          'downloadFile $urlPath -> $destination failed: ${response.statusMessage}.');
    }
  }

  Future<ServerData> postServer(MLModel model, bool startFresh) async {
    PostServerData body = PostServerData(id: model.id, start_fresh: startFresh);
    Response response = await dio.post('$url/train/server', data: body.toMap());
    return ServerDataMapper.fromMap(response.data);
  }

  Future<Response> fitInsTelemetry(FitInsTelemetryData body) async {
    return await dio.post('$url/telemetry/fit_ins', data: body.toJson());
  }

  Future<Response> evaluateInsTelemetry(EvaluateInsTelemetryData body) async {
    return await dio.post('$url/telemetry/evaluate_ins', data: body.toJson());
  }
}

// Always change together with Django `train.serializers.PostAdvertisedDataSerializer`.
@MappableClass()
class PostAdvertisedData with PostAdvertisedDataMappable {
  final String data_type;
  final bool tflite;
  final bool coreml;

  PostAdvertisedData({
    required this.data_type,
    this.tflite = true,
    this.coreml = false,
  });
}

// Always change together with Django `train.data.ServerData`.
@MappableClass()
class ServerData with ServerDataMappable {
  final String status;
  final int? session_id;
  final int? port;

  ServerData({
    required this.status,
    this.session_id,
    this.port,
  });
}

// Always change together with Django `train.serializers.PostServerDataSerializer`.
@MappableClass()
class PostServerData with PostServerDataMappable {
  final int id;
  final bool start_fresh;

  PostServerData({
    required this.id,
    required this.start_fresh,
  });
}

@MappableClass()
class FitInsTelemetryData with FitInsTelemetryDataMappable {
  final int device_id;
  final int session_id;
  final int start;
  final int end;

  FitInsTelemetryData({
    required this.device_id,
    required this.session_id,
    required this.start,
    required this.end,
  });
}

@MappableClass()
class EvaluateInsTelemetryData with EvaluateInsTelemetryDataMappable {
  final int device_id;
  final int session_id;
  final int start;
  final int end;
  final double loss;
  final double accuracy;
  final int test_size;

  EvaluateInsTelemetryData({
    required this.device_id,
    required this.session_id,
    required this.start,
    required this.end,
    required this.loss,
    required this.accuracy,
    required this.test_size,
  });
}

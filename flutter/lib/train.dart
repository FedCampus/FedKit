import 'dart:io';

import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/log.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:fed_kit/train_state.dart';

class Train {
  bool _telemetry = false;
  bool get telemetry => _telemetry;
  int _deviceId = 0;
  int get deviceId => _deviceId;
  late BackendClient _client;
  TrainState _state = Initialized();
  int? _sessionId;

  Train(String backendUrl) {
    _client = BackendClient(backendUrl);
  }

  void enableTelemetry(int deviceId) {
    _telemetry = true;
    _deviceId = deviceId;
  }

  void disableTelemetry() {
    _deviceId = 0;
    _telemetry = false;
  }

  Future<String> prepareModel(String dataType) => switch (_state) {
        Initialized() || WithModel() => _prepareModel(dataType),
        _ => throw Exception('`prepareModel` called with $_state'),
      };

  Future<String> _prepareModel(String dataType) async {
    final model = await _advertisedModel(dataType);
    final modelDir = await getModelDir(model);
    await downloadModelFile(modelDir);
    return modelDir;
  }

  Future<TFLiteModel> advertisedModel(String dataType) async =>
      switch (_state) {
        Initialized() || WithModel() => _advertisedModel(dataType),
        _ => throw Exception('`advertisedModel` called with $_state'),
      };

  Future<TFLiteModel> _advertisedModel(String dataType) async {
    final model =
        await _client.advertisedModel(PostAdvertisedData(data_type: dataType));
    logger.d('Advertised model: $model.');
    _state = WithModel(model);
    return model;
  }

  Future<void> downloadModelFile(String modelDir) => switch (_state) {
        WithModel state => _downloadModelFile(modelDir, state.model),
        _ => throw Exception('`downloadModelFile` called with $_state'),
      };

  Future<void> _downloadModelFile(String modelDir, TFLiteModel model) async {
    if (await Directory(modelDir).exists()) {
      logger.d('Skipping already downloaded model ${model.name}');
      return;
    }
    final fileUrl = model.file_path;
    await _client.downloadFile(fileUrl, modelDir);
    logger.d('Downloaded ${model.name}: $fileUrl -> $modelDir.');
  }

  Future<ServerData> getServerInfo({bool startFresh = false}) =>
      switch (_state) {
        WithModel state => _getServerInfo(state.model, startFresh),
        _ => throw Exception('`getServerInfo` called with $_state'),
      };

  Future<ServerData> _getServerInfo(TFLiteModel model, bool startFresh) async {
    final serverData = await _client.postServer(model, startFresh);
    _sessionId = serverData.session_id;
    logger.d('Server info: $serverData.');
    return serverData;
  }

  void mlClientReady(MLClient mlClient) => switch (_state) {
        WithModel state => _state = Prepared(state.model, mlClient),
        _ => throw Exception('`mlClientReady` called with $_state'),
      };
}

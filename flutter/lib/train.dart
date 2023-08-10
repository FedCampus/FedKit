import 'dart:io';

import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/flower_service.dart';
import 'package:fed_kit/log.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:fed_kit/train_state.dart';
import 'package:grpc/grpc.dart';

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

  Future<(TFLiteModel, String)> prepareModel(String dataType) => switch (_state) {
        Initialized() || WithModel() => _prepareModel(dataType),
        _ => throw Exception('`prepareModel` called with $_state'),
      };

  Future<(TFLiteModel, String)> _prepareModel(String dataType) async {
    final model = await _advertisedModel(dataType);
    final modelDir = await getModelDir(model);
    await downloadModelFile(modelDir);
    return (model, modelDir);
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

  /// After ML client is ready.
  Future<void> prepare(MLClient mlClient, String address, int port,
          {ChannelOptions channelOptions = defaultChannelOptions}) =>
      switch (_state) {
        WithModel state =>
          _prepare(mlClient, state, address, port, channelOptions),
        _ => throw Exception('`mlClientReady` called with $_state'),
      };

  Future<void> _prepare(MLClient mlClient, WithModel state, String address,
      int port, ChannelOptions options) async {
    if (!await mlClient.ready()) {
      throw Exception('`mlClient` not ready');
    }
    final channel = ClientChannel(
      address,
      port: port,
      options: options,
    );

    _state = Prepared(state.model, mlClient, channel);
  }

  FlowerService start(Function(String) onInfo) => switch (_state) {
        Prepared state => _start(onInfo, state),
        _ => throw Exception('`start` called with $_state'),
      };

  FlowerService _start(Function(String) onInfo, Prepared state) {
    final model = state.model;
    final flowerService =
        FlowerService(state.channel, this, state.mlClient, model, onInfo)
          ..run();
    logger.d('Training for ${model.name} started.');
    _state = Training(model, flowerService);
    return flowerService;
  }

  Future<void> fitInsTelemetry(DateTime start, DateTime end) async {
    checkTelemetryEnabled();
    final body = FitInsTelemetryData(
        device_id: deviceId,
        session_id: _sessionId!,
        start: start.millisecond,
        end: end.microsecond);
    await _client.fitInsTelemetry(body);
    logger.d('Train telemetry: sent FitIns.');
  }

  Future<void> evaluateInsTelemetry(DateTime start, DateTime end, double loss,
      double accuracy, int testSize) async {
    checkTelemetryEnabled();
    final body = EvaluateInsTelemetryData(
        device_id: deviceId,
        session_id: _sessionId!,
        start: start.millisecond,
        end: end.microsecond,
        loss: loss,
        accuracy: accuracy,
        test_size: testSize);
    await _client.evaluateInsTelemetry(body);
    logger.d('Train telemetry: sent EvaluateIns.');
  }

  void checkTelemetryEnabled() {
    if (!telemetry || _sessionId == null) {
      throw Exception('Telemetry disabled');
    }
  }
}

const defaultChannelOptions =
    ChannelOptions(credentials: ChannelCredentials.insecure());

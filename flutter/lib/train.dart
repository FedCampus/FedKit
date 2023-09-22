import 'dart:io';

import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/flower_service.dart';
import 'package:fed_kit/log.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/ml_model.dart';
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

  Future<(MLModel, String)> prepareModel(String dataType) => switch (_state) {
        Initialized() || WithModel() => _prepareModel(dataType),
        _ => throw Exception('`prepareModel` called with $_state'),
      };

  Future<(MLModel, String)> _prepareModel(String dataType) async {
    final model = await _whichModel(dataType);
    final (modelUrl, modelDir) = await model.urlAndDir;
    await downloadModelFile(modelUrl, modelDir);
    return (model, modelDir);
  }

  Future<MLModel> whichModel(String dataType) async => switch (_state) {
        Initialized() || WithModel() => _whichModel(dataType),
        _ => throw Exception('`advertisedModel` called with $_state'),
      };

  Future<MLModel> _whichModel(String dataType) async {
    final model = await _client.whichModel(PostAdvertisedData(
        data_type: dataType, tflite: !Platform.isIOS, coreml: Platform.isIOS));
    logger.d('Advertised model: $model.');
    _state = WithModel(model);
    return model;
  }

  Future<void> downloadModelFile(String modelUrl, String modelDir) =>
      switch (_state) {
        WithModel state => _downloadModelFile(modelUrl, modelDir, state.model),
        _ => throw Exception('`downloadModelFile` called with $_state'),
      };

  Future<void> _downloadModelFile(
      String modelUrl, String modelDir, MLModel model) async {
    if (await Directory(modelDir).exists()) {
      logger.d('Skipping already downloaded model ${model.name}');
      return;
    }
    await _client.downloadFile(modelUrl, modelDir);
    logger.d('Downloaded ${model.name}: $modelUrl -> $modelDir.');
  }

  Future<ServerData> getServerInfo({bool startFresh = false}) =>
      switch (_state) {
        WithModel state => _getServerInfo(state.model, startFresh),
        _ => throw Exception('`getServerInfo` called with $_state'),
      };

  Future<ServerData> _getServerInfo(MLModel model, bool startFresh) async {
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

  Stream<String> start() => switch (_state) {
        Prepared state => _start(state),
        _ => throw Exception('`start` called with $_state'),
      };

  Stream<String> _start(Prepared state) {
    final model = state.model;
    final flowerService =
        FlowerService(state.channel, this, state.mlClient, model);
    final stream = flowerService.run();
    logger.d('Training for ${model.name} started.');
    _state = Training(model, flowerService);
    return stream;
  }

  Future<void> fitInsTelemetry(DateTime start, DateTime end) async {
    checkTelemetryEnabled();
    final body = FitInsTelemetryData(
        device_id: deviceId,
        session_id: _sessionId!,
        start: start.millisecondsSinceEpoch,
        end: end.millisecondsSinceEpoch);
    await _client.fitInsTelemetry(body);
    logger.d('Train telemetry: sent FitIns.');
  }

  Future<void> evaluateInsTelemetry(DateTime start, DateTime end, double loss,
      double accuracy, int testSize) async {
    checkTelemetryEnabled();
    final body = EvaluateInsTelemetryData(
        device_id: deviceId,
        session_id: _sessionId!,
        start: start.millisecondsSinceEpoch,
        end: end.millisecondsSinceEpoch,
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

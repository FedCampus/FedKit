/// A portion of this file is copied from
// https://github.com/Anthrapper/flutterFlower/blob/main/lib/app/flower/flower_client.dart
// and
// https://github.com/Anthrapper/flutterFlower/blob/main/lib/app/flower/flower_base.dart.
import 'dart:async';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:fed_kit/log.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/ml_model.dart';
import 'package:fed_kit/train.dart';
import 'package:fed_kit/transport.pbgrpc.dart';
import 'package:fixnum/fixnum.dart';
import 'package:flutter/foundation.dart';
import 'package:grpc/grpc.dart';

class FlowerService {
  final _msgStreamCtl = StreamController<ClientMessage>();
  final _infoStreamCtl = StreamController<String>.broadcast();
  bool _done = false;
  final jobs = <Future<void>>[];
  final ClientChannel channel;
  final Train train;
  final MLClient mlClient;
  final MlModel model;
  late StreamSubscription<ServerMessage> _streamSub;

  FlowerService(this.channel, this.train, this.mlClient, this.model);

  /// Start running this service.
  /// Returns a stream of info.
  Stream<String> run() {
    _streamSub = FlowerServiceClient(channel)
        .join(_msgStreamCtl.stream)
        .listen(_handleMessage, onError: (e, s) {
      _logErr(e, s);
      _infoStreamCtl.addError(e, s);
      close();
    }, onDone: close, cancelOnError: true);
    return _infoStreamCtl.stream;
  }

  /// Stream of information produced.
  Stream<String> get infoStream => _infoStreamCtl.stream;

  Future<void> _handleMessage(ServerMessage message) async {
    ClientMessage? response;
    if (message.hasGetParametersIns()) {
      response = await _handleGetParameters(message);
    } else if (message.hasFitIns()) {
      response = await _handleFitIns(message);
    } else if (message.hasEvaluateIns()) {
      response = await _handleEvaluateIns(message);
    } else if (message.hasReconnectIns()) {
      _logDebug('Handling reconnectIns');
      await close();
    } else {
      throw Exception('Unknown message type: $message');
    }
    if (response != null) {
      _sendMessage(response);
    }
  }

  Future<ClientMessage> _handleGetParameters(ServerMessage message) async {
    _logDebug('Handling GetParameters');
    _infoStreamCtl.add("Handling GetParameters message from the server.");
    return weightsAsProto(await mlClient.getParameters());
  }

  Future<ClientMessage> _handleFitIns(ServerMessage message) async {
    _logDebug('Handling FitIns');
    _infoStreamCtl.add('Handling FitIns message from the server.');
    final start = train.telemetry ? DateTime.now() : null;
    final layers =
        message.fitIns.parameters.tensors.map(Uint8List.fromList).toList();
    final epochConfig = message.fitIns.config['local_epochs'];
    final epochs = epochConfig?.sint64.toInt() ?? 1;
    await mlClient.updateParameters(layers);
    await mlClient.fit(
        epochs: epochs,
        onLoss: (losses) =>
            _infoStreamCtl.add('Average loss: ${losses.average}'));
    if (start != null) {
      final end = DateTime.now();
      final job = _spawnLogErr(() => train.fitInsTelemetry(start, end));
      jobs.add(job);
    }
    return fitResAsProto(
        await mlClient.getParameters(), await mlClient.trainingSize);
  }

  Future<ClientMessage> _handleEvaluateIns(ServerMessage message) async {
    _logDebug('Handling EvaluateIns message from the server');
    _infoStreamCtl.add('Handling EvaluateIns');
    final start = train.telemetry ? DateTime.now() : null;
    final layers =
        message.evaluateIns.parameters.tensors.map(Uint8List.fromList).toList();
    await mlClient.updateParameters(layers);
    final (loss, accuracy) = await mlClient.evaluate();
    _infoStreamCtl.add('Test accuracy after this round: $accuracy.');
    final testSize = await mlClient.testSize;
    if (start != null) {
      final end = DateTime.now();
      final job = _spawnLogErr(() =>
          train.evaluateInsTelemetry(start, end, loss, accuracy, testSize));
      jobs.add(job);
    }
    return evaluateResAsProto(loss, testSize);
  }

  void _sendMessage(ClientMessage message) {
    _msgStreamCtl.add(message);
  }

  Future<void> _spawnLogErr(Future<void> Function() call) async {
    try {
      await call();
    } catch (err, stackTrace) {
      _logErr(err, stackTrace);
    }
  }

  void _logErr(Object err, StackTrace stackTrace) {
    logger.e('handleMessage: $err, $stackTrace.');
  }

  void _logDebug(String message) {
    logger.d('FlowerService: $message.');
  }

  /// Shut down this service.
  Future<void> close() async {
    if (done) {
      logger.w('FlowerService: second exit.');
      return;
    }
    await _streamSub.cancel();
    await channel.shutdown();
    for (final job in jobs) {
      await job;
    }
    _logDebug('exit');
    _done = true;
    _infoStreamCtl.close();
  }

  bool get done => _done;
}

ClientMessage weightsAsProto(List<Uint8List> weights) {
  Parameters p = Parameters()
    ..tensors.addAll(weights)
    ..tensorType = "ND";
  ClientMessage_GetParametersRes res = ClientMessage_GetParametersRes()
    ..parameters = p;
  return ClientMessage()..getParametersRes = res;
}

ClientMessage fitResAsProto(List<Uint8List> weights, int trainingSize) {
  Parameters p = Parameters()
    ..tensors.addAll(weights)
    ..tensorType = 'ND';
  ClientMessage_FitRes res = ClientMessage_FitRes()
    ..parameters = p
    ..numExamples = Int64(trainingSize);
  return ClientMessage()..fitRes = res;
}

ClientMessage evaluateResAsProto(double accuracy, int testSize) {
  ClientMessage_EvaluateRes res = ClientMessage_EvaluateRes()
    ..loss = accuracy
    ..numExamples = Int64(testSize);
  return ClientMessage()..evaluateRes = res;
}

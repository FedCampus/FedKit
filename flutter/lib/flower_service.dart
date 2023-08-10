/// A portion of this file is copied from
// https://github.com/Anthrapper/flutterFlower/blob/main/lib/app/flower/flower_client.dart
// and
// https://github.com/Anthrapper/flutterFlower/blob/main/lib/app/flower/flower_base.dart.
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:fed_kit/helpers.dart';
import 'package:fed_kit/log.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:fed_kit/train.dart';
import 'package:fed_kit/transport.pbgrpc.dart';
import 'package:fixnum/fixnum.dart';
import 'package:flutter/foundation.dart';
import 'package:grpc/grpc.dart';

class FlowerService {
  final streamController = StreamController<ClientMessage>();
  bool _done = false;
  final jobs = <Future<void>>[];
  final _listeners = <Completer<void>>[];
  final ClientChannel channel;
  final Train train;
  final MLClient mlClient;
  final TFLiteModel model;
  final Function(String) onInfo;
  late StreamSubscription<ServerMessage> _streamSub;

  FlowerService(
      this.channel, this.train, this.mlClient, this.model, this.onInfo);

  run() {
    _streamSub = FlowerServiceClient(channel)
        .join(streamController.stream)
        .listen(_handleMessage, onError: (e, s) async {
      _logErr(e, s);
      await close();
    }, onDone: close, cancelOnError: true);
  }

  Future<void> wait() async {
    if (done) {
      return;
    }
    final completer = Completer();
    _listeners.add(completer);
    if (done) {
      return;
    }
    await completer.future;
  }

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
    onInfo("Handling GetParameters message from the server.");
    return weightsAsProto(await mlClient.getParameters());
  }

  Future<ClientMessage> _handleFitIns(ServerMessage message) async {
    _logDebug('Handling FitIns');
    onInfo('Handling FitIns message from the server.');
    final start = train.telemetry ? DateTime.now() : null;
    final layers =
        message.fitIns.parameters.tensors.map(Uint8List.fromList).toList();
    assertEqual(layers.length, model.layers_sizes.length);
    final epochConfig = message.fitIns.config['local_epochs'];
    final epochs = epochConfig?.sint64.toInt() ?? 1;
    await mlClient.updateParameters(layers);
    await mlClient.fit(
        epochs: epochs,
        onLoss: (losses) => onInfo('Average loss: ${losses.average}'));
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
    onInfo('Handling EvaluateIns');
    final start = train.telemetry ? DateTime.now() : null;
    final layers =
        message.evaluateIns.parameters.tensors.map(Uint8List.fromList).toList();
    assertEqual(layers.length, model.layers_sizes.length);
    await mlClient.updateParameters(layers);
    final lossAccuracy = await mlClient.evaluate();
    final loss = lossAccuracy[0];
    final accuracy = lossAccuracy[1];
    onInfo('Test accuracy after this round: $accuracy.');
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
    streamController.add(message);
  }

  Future<void> _spawnLogErr(Function() call) => Isolate.run(() {
        try {
          call();
        } catch (err, stackTrace) {
          _logErr(err, stackTrace);
        }
      });

  void _logErr(Object err, StackTrace stackTrace) {
    logger.e('handleMessage: $err, $stackTrace.');
  }

  void _logDebug(String message) {
    logger.d('FlowerService: $message.');
  }

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
    for (final completer in _listeners) {
      completer.complete();
    }
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

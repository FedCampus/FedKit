import 'dart:typed_data';

import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit_client/main.dart';
import 'package:flutter/services.dart';

class Cifar10MLClient extends MLClient {
  final callChannel = const MethodChannel('fed_kit_client_cifar10_ml_client');
  final logChannel = const EventChannel('fed_kit_client_cifar10_ml_client_log');
  bool _listening = false;
  Function(List<double> p1)? _onLoss;

  Future<String?> getPlatformVersion() =>
      callChannel.invokeMethod<String>('getPlatformVersion');

  @override
  Future<(double, double)> evaluate() async {
    final lossAccuracy =
        await callChannel.invokeMethod<Float32List>('evaluate');
    return (lossAccuracy![0], lossAccuracy[1]);
  }

  @override
  Future<void> fit(
      {int epochs = 1,
      int batchSize = 32,
      Function(List<double> p1)? onLoss}) async {
    _onLoss = onLoss;
    ensureListening();
    await callChannel.invokeMethod('fit', {
      'epochs': epochs,
      'batchSize': batchSize,
    });
  }

  @override
  Future<List<Uint8List>> getParameters() async {
    final params = await callChannel.invokeMethod<List>('getParameters');
    return params!.cast<Uint8List>().toList();
  }

  @override
  Future<bool> ready() async {
    final params = await callChannel.invokeMethod<bool>('ready');
    return params!;
  }

  @override
  Future<int> get testSize async {
    final size = await callChannel.invokeMethod<int>('testSize');
    return size!;
  }

  @override
  Future<int> get trainingSize async {
    final size = await callChannel.invokeMethod<int>('trainingSize');
    return size!;
  }

  @override
  Future<void> updateParameters(List<Uint8List> parameters) async {
    await callChannel
        .invokeMethod('updateParameters', {'parameters': parameters});
  }

  Future<void> initML(
      String modelDir, List<int> layersSizes, int partitionId) async {
    final result= await callChannel.invokeMethod('initML', {
      'modelDir': modelDir,
      'layersSizes': layersSizes,
      'partitionId': partitionId
    });
    logger.d("initML: $result.");
  }

  void ensureListening() {
    if (_listening) return;
    logChannel.receiveBroadcastStream().listen(_callOnLoss);
    _listening = true;
  }

  void _callOnLoss(loss) {
    _onLoss?.call(loss.cast<double>());
  }
}

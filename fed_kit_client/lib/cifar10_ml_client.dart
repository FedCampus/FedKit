import 'dart:typed_data';

import 'package:fed_kit/ml_client.dart';

class Cifar10MLClient extends MLClient {
  @override
  Future<List<double>> evaluate() {
    // TODO: implement evaluate
    throw UnimplementedError();
  }

  @override
  Future<void> fit(
      {int epochs = 1, int batchSize = 32, Function(List<double> p1)? onLoss}) {
    // TODO: implement fit
    throw UnimplementedError();
  }

  @override
  Future<List<Uint8List>> getParameters() {
    // TODO: implement getParameters
    throw UnimplementedError();
  }

  @override
  Future<bool> ready() {
    // TODO: implement ready
    throw UnimplementedError();
  }

  @override
  // TODO: implement testSize
  Future<int> get testSize => throw UnimplementedError();

  @override
  // TODO: implement trainingSize
  Future<int> get trainingSize => throw UnimplementedError();

  @override
  Future<void> updateParameters(List<Uint8List> parameters) {
    // TODO: implement updateParameters
    throw UnimplementedError();
  }
}

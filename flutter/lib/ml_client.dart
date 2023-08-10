import 'dart:typed_data';

abstract class MLClient {
  Future<List<Uint8List>> getParameters();

  Future<void> updateParameters(List<Uint8List> parameters);

  Future<bool> ready();

  Future<void> fit(
      {int epochs = 1, int batchSize = 32, Function(List<double>)? onLoss});

  Future<int> get trainingSize;

  Future<int> get testSize;

  Future<List<double>> evaluate();
}

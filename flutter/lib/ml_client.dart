import 'dart:typed_data';

abstract class MLClient {
  Future<List<Uint8List>> getParameters();

  Future<void> updateParameters(List<Uint8List> parameters);

  Future<void> fit({int epochs = 1, int batchSize = 32});

  Future<List<double>> evaluate();
}

import 'dart:typed_data';

import 'package:fed_kit/ml_client.dart';

class FakeMLClient extends MLClient {
  List<Uint8List> _parameters = [];

  @override
  Future<List<double>> evaluate() async => [0.1, 0.9];

  @override
  Future<void> fit({int epochs = 1, int batchSize = 32}) async {}

  @override
  Future<List<Uint8List>> getParameters() async => _parameters;

  @override
  Future<void> updateParameters(List<Uint8List> parameters) async {
    _parameters = parameters;
  }
}

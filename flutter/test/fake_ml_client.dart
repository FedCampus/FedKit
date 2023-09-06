import 'dart:typed_data';

import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/ml_model.dart';

class FakeMLClient extends MLClient {
  List<Uint8List>? _parameters;

  List<Uint8List> get parameters {
    _parameters ??=
        model.tflite_layers!.map((size) => Uint8List(size)).toList();
    return _parameters!;
  }

  final MLModel model;

  FakeMLClient(this.model);

  @override
  Future<(double, double)> evaluate() async => (0.1, 0.9);

  @override
  Future<bool> ready() async => true;

  @override
  Future<void> fit(
      {int epochs = 1,
      int batchSize = 32,
      Function(List<double>)? onLoss}) async {}

  @override
  Future<List<Uint8List>> getParameters() async => parameters;

  @override
  Future<int> get trainingSize async => 10;

  @override
  Future<int> get testSize async => 10;

  @override
  Future<void> updateParameters(List<Uint8List> parameters) async {
    _parameters = parameters;
  }
}

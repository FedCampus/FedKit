import 'package:fed_kit/tflite_model.dart';

sealed class TrainState {}

class Initialized extends TrainState {}

class WithModel extends TrainState {
  final TFLiteModel model;

  WithModel(this.model);
}

/// The platform side has a `FlowerClient` ready.
class Prepared extends TrainState {
  final TFLiteModel model;

  Prepared(this.model);
}

class Training extends TrainState {
  final TFLiteModel model;
  // TODO: flower service object.

  Training(this.model);
}

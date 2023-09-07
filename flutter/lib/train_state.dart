import 'package:fed_kit/flower_service.dart';
import 'package:fed_kit/ml_client.dart';
import 'package:fed_kit/ml_model.dart';
import 'package:grpc/grpc.dart';

sealed class TrainState {}

class Initialized extends TrainState {}

class WithModel extends TrainState {
  final MLModel model;

  WithModel(this.model);
}

/// The platform side has a `FlowerClient` ready.
class Prepared extends TrainState {
  final MLModel model;
  final MLClient mlClient;
  final ClientChannel channel;

  Prepared(this.model, this.mlClient, this.channel);
}

class Training extends TrainState {
  final MLModel model;
  final FlowerService flowerService;

  Training(this.model, this.flowerService);
}

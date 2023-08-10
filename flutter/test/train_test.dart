@Skip('Launch a backend and test with `--run-skipped` for integration test.')
import 'dart:io';

import 'package:fed_kit/train.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';

import '../test/backend_client_test.dart';
import 'train_test_helpers.dart';

void main() {
  CustomBindings();
  PathProviderPlatform.instance = FakePathProviderPlatform();

  final train = Train(exampleBackendUrl);

  test('alter telemetry', () async {
    train.enableTelemetry(1);
    expect(train.telemetry, true);
    expect(train.deviceId, 1);
    train.disableTelemetry();
    expect(train.telemetry, false);
  });

  test('prepare training', () async {
    final modelFile = await train.prepareModel(dataType);
    expect(
        modelFile, '$kApplicationDocumentsPath/models/CIFAR10/cifar10.tflite');
    Directory(kApplicationDocumentsPath).deleteSync(recursive: true);
  });
}

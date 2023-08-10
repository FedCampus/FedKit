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

  test('prepare training', () async {
    final modelFile = await train.prepareModel(dataType);
    expect(
        modelFile, '$kApplicationDocumentsPath/models/CIFAR10/cifar10.tflite');
    Directory(kApplicationDocumentsPath).deleteSync(recursive: true);
  });
}

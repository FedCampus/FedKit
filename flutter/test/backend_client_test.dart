@Skip('Launch a backend and test with `--run-skipped` for integration test.')
import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:flutter_test/flutter_test.dart';

const url = 'http://0:8000';
const dataType = 'CIFAR10_32x32x3';

void main() {
  const client = BackendClient(url);

  test('ask backend for advertised model', () async {
    final actual = await client
        .advertisedModel(const PostAdvertisedData(data_type: dataType));
    const expected = TFLiteModel(
        id: 1,
        name: 'CIFAR10',
        file_path: '/static/cifar10.tflite',
        layers_sizes: [1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40]);
    expect(actual, expected);
  });
}

@Skip('Launch a backend and test with `--run-skipped` for integration test.')
import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/tflite_model.dart';
import 'package:flutter_test/flutter_test.dart';

const exampleBackendUrl = 'http://0:8000';
const exampleFlowerAddress = '0';
const dataType = 'CIFAR10_32x32x3';
final expectedModel = TFLiteModel(
    id: 1,
    name: 'CIFAR10',
    file_path: '/static/cifar10.tflite',
    layers_sizes: [1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40]);
const expectedPort = 8080;
final fitInsTelemetry =
    FitInsTelemetryData(device_id: 1, session_id: 1, start: 0, end: 1);
final evaluateInsTelemetry = EvaluateInsTelemetryData(
    device_id: 1,
    session_id: 1,
    start: 0,
    end: 1,
    loss: 0.1,
    accuracy: 0.8,
    test_size: 1);

void main() {
  const client = BackendClient(exampleBackendUrl);

  test('ask backend for advertised model', () async {
    final actual =
        await client.advertisedModel(PostAdvertisedData(data_type: dataType));
    expect(actual, expectedModel);
  });

  test('ask backend for Flower server', () async {
    var actual = await client.postServer(expectedModel, false);
    expect(actual.port, expectedPort);
    actual = await client.postServer(expectedModel, false);
    expect(actual.port, expectedPort);
    expect(actual.status, 'started');
    actual = await client.postServer(expectedModel, true);
    expect(actual.port, null);
    expect(actual.status, 'started_non_fresh');
  });

  test('send backend telemetry', () async {
    var response = await client.fitInsTelemetry(fitInsTelemetry);
    expect(response.statusCode, 200);
    response = await client.evaluateInsTelemetry(evaluateInsTelemetry);
    expect(response.statusCode, 200);
  });
}

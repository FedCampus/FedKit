import 'dart:io';

import 'package:app_set_id/app_set_id.dart';
import 'package:archive/archive.dart';
import 'package:fed_kit/backend_client.dart';
import 'package:fed_kit/train.dart';
import 'package:fed_kit_client/mnist_ml_client.dart';
import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import 'package:path_provider/path_provider.dart';

final logger = Logger();

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _platformVersion = 'Unknown';
  final _mlClient = MnistMLClient();
  var canPrepare = true;
  var canTrain = false;
  var startFresh = false;
  late Train train;

  @override
  void initState() {
    super.initState();
    initPlatformState();
  }

  Future<void> initPlatformState() async {
    String platformVersion;
    try {
      platformVersion =
          await _mlClient.getPlatformVersion() ?? 'Unknown platform version';
    } on PlatformException {
      platformVersion = 'Failed to get platform version.';
    }

    // If the widget was removed from the tree while the asynchronous platform
    // message was in flight, we want to discard the reply rather than calling
    // setState to update our non-existent appearance.
    if (!mounted) return;

    setState(() {
      _platformVersion = platformVersion;
      appendLog('Running on: $_platformVersion.');
    });
  }

  final logs = [const Text('Logs will be shown here.')];
  final clientPartitionIdController = TextEditingController();
  final flServerIPController = TextEditingController();
  final flServerPortController = TextEditingController();
  final scrollController = ScrollController();

  appendLog(String message) {
    logger.d('appendLog: $message');
    setState(() {
      logs.add(Text(message));
    });
  }

  prepare() async {
    int partitionId;
    try {
      partitionId = int.parse(clientPartitionIdController.text);
    } catch (e) {
      return appendLog('Invalid client partition id!');
    }
    Uri host;
    try {
      host = Uri.parse('http://${flServerIPController.text}');
      if (!host.hasEmptyPath || host.host.isEmpty || host.hasPort) {
        throw Exception();
      }
    } catch (e) {
      return appendLog('Invalid backend server host!');
    }
    Uri backendUrl;
    int backendPort;
    try {
      backendPort = int.parse(flServerPortController.text);
      backendUrl = host.replace(port: backendPort);
    } catch (e) {
      return appendLog('Invalid backend server port!');
    }

    canPrepare = false;
    appendLog(
        'Connecting with Partition ID: $partitionId, Server IP: $host, Port: $backendPort');

    try {
      await _prepare(partitionId, host, backendUrl);
    } on PlatformException catch (error, stacktrace) {
      canPrepare = true;
      appendLog('Request failed: ${error.message}.');
      logger.e('$error\n$stacktrace.');
    } catch (error, stacktrace) {
      canPrepare = true;
      appendLog('Request failed: $error.');
      logger.e(stacktrace);
    }
  }

  _prepare(int partitionId, Uri host, Uri backendUrl) async {
    final downloadTask = downloadDataSet();
    train = Train(backendUrl.toString());
    final id = await deviceId();
    logger.d('Device ID: $id');
    train.enableTelemetry(id);
    final (model, modelDir) = await train.prepareModel(dataType);
    appendLog('Prepared model ${model.name}.');
    final serverData = await train.getServerInfo(startFresh: startFresh);
    if (serverData.port == null) {
      throw Exception(
          'Flower server port not available", "status ${serverData.status}');
    }
    appendLog(
        'Ready to connected to Flower server on port ${serverData.port}.');
    final layersSizes =
        Platform.isIOS ? model.coreml_layers! : model.tflite_layers!;
    final dataDir = await downloadTask;
    logger.d('Data directory: $dataDir');
    await _mlClient.initML(dataDir, modelDir, layersSizes, partitionId);
    appendLog('Prepared ML client and loaded dataset.');
    await train.prepare(_mlClient, host.host, serverData.port!);
    canTrain = true;
    appendLog('Ready to train.');
  }

  startTrain() async {
    try {
      train.start().listen(appendLog, onDone: () {
        canPrepare = true;
        appendLog('Training done.');
      }, onError: (e) {
        canPrepare = true;
        appendLog('Training failed: $e.');
      }, cancelOnError: true);
      canTrain = false;
      appendLog('Started training.');
    } on PlatformException catch (error, stacktrace) {
      canTrain = true;
      appendLog('Training failed: ${error.message}.');
      logger.e('$error\n$stacktrace.');
    } catch (error, stacktrace) {
      canTrain = true;
      appendLog('Failed to start training: $error.');
      logger.e(stacktrace);
    }
  }

  @override
  Widget build(BuildContext context) {
    final children = [
      TextFormField(
        controller: clientPartitionIdController,
        decoration: const InputDecoration(
          labelText: 'Client Partition ID (1-10)',
          filled: true,
        ),
        keyboardType: TextInputType.number,
      ),
      TextFormField(
        controller: flServerIPController,
        decoration: const InputDecoration(
          labelText: 'Backend Server Host',
          filled: true,
        ),
        keyboardType: TextInputType.text,
      ),
      TextFormField(
        controller: flServerPortController,
        decoration: const InputDecoration(
          labelText: 'Backend Server Port',
          filled: true,
        ),
        keyboardType: TextInputType.number,
      ),
      Row(
        children: [
          Checkbox(
              value: startFresh,
              onChanged: (checked) {
                setState(() => startFresh = checked!);
              }),
          const Text('Start Fresh')
        ],
      ),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        ElevatedButton(
          onPressed: canPrepare ? prepare : null,
          child: const Text('Prepare'),
        ),
        ElevatedButton(
          onPressed: canTrain ? startTrain : null,
          child: const Text('Train'),
        ),
      ]),
      Expanded(
        child: ListView.builder(
          controller: scrollController,
          reverse: true,
          padding: const EdgeInsets.only(
              top: 16.0, bottom: 32.0, left: 12.0, right: 12.0),
          itemCount: logs.length,
          itemBuilder: (context, index) => logs[logs.length - index - 1],
        ),
      ),
    ];

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('FedKit example app'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: children,
        ),
      ),
    );
  }
}

Future<int> deviceId() async => (await AppSetId().getIdentifier()).hashCode;

Future<String> downloadDataSet() async {
  const url =
      'https://gitlab.com/SichangHe/notes/uploads/293000f0e66d63ca2ce4d6082f8eb632/MNIST_data.zip';
  final tempDir = '${(await getApplicationDocumentsDirectory()).path}/MNIST/';
  final zipFile = File('$tempDir/MNIST_data.zip');
  if (!await zipFile.exists()) {
    await dio.download(url, zipFile.path);
  }
  final trainFile = File('$tempDir/MNIST_train.csv');
  final testFile = File('$tempDir/MNIST_test.csv');
  if (await trainFile.exists() && await testFile.exists()) {
    return tempDir;
  }
  final archive = ZipDecoder().decodeBytes(await zipFile.readAsBytes());
  for (final file in archive) {
    final fileName = '$tempDir/${file.name}';
    if (file.isFile) {
      final outFile = File(fileName);
      await outFile.create(recursive: true);
      await outFile.writeAsBytes(file.content);
    }
  }
  return tempDir;
}

const dataType = 'MNIST_28x28x1';

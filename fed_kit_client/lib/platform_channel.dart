import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class PlatformChannel {
  @visibleForTesting
  final methodChannel = const MethodChannel('fed_kit_flutter');

  Future<String?> getPlatformVersion() async {
    return await methodChannel.invokeMethod<String>('getPlatformVersion');
  }

  Future<int?> connect(int partitionId, Uri host, Uri backendUrl,
      {bool startFresh = false}) async {
    return await methodChannel.invokeMethod<int>('connect', {
      'partitionId': partitionId,
      'host': host.host,
      'backendUrl': backendUrl.toString(),
      'startFresh': startFresh,
    });
  }

  Future<void> train() async {
    await methodChannel.invokeMethod('train');
  }
}

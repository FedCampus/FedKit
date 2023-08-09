class Train {
  bool _telemetry = false;
  bool get telemetry => _telemetry;
  int _deviceId = 0;
  int get deviceId => _deviceId;

  void enableTelemetry(int deviceId) {
    _telemetry = true;
    _deviceId = deviceId;
  }

  void disableTelemetry() {
    _deviceId = 0;
    _telemetry = false;
  }
}

typedef WebBleValueCallback = void Function(List<int> value);
typedef WebBleDisconnectCallback = void Function();
typedef WebBleStatusCallback = void Function(String message);

class WebBleAdapter {
  String get deviceName => 'No Device';

  Future<bool> connect({
    required String serviceUuid,
    required String characteristicUuid,
    required WebBleValueCallback onValue,
    required WebBleDisconnectCallback onDisconnected,
    required WebBleStatusCallback onStatus,
  }) async {
    onStatus(
      'Web Bluetooth is unavailable on this platform. '
      'Use a supported browser or native BLE application.',
    );

    return false;
  }

  Future<void> disconnect() async {}
}
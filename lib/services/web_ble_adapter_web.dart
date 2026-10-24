@JS()
library;

import 'dart:async';
import 'dart:js_interop';
import 'dart:convert';

import 'web_ble_adapter_stub.dart';

@JS('window.esp32Ble.connect')
external JSPromise<JSBoolean> _connectBle(
  JSString serviceUuid,
  JSString characteristicUuid,
  JSFunction onData,
  JSFunction onStatus,
);

@JS('window.esp32Ble.disconnect')
external JSAny? _disconnectBle();

@JS('window.esp32Ble.deviceName')
external JSString get _bleDeviceName;

class WebBleAdapter {
  String _deviceName = 'No Device';
  bool _connected = false;

  String get deviceName => _deviceName;
  bool get isConnected => _connected;

  Future<bool> connect({
    required String serviceUuid,
    required String characteristicUuid,
    required WebBleValueCallback onValue,
    required WebBleDisconnectCallback onDisconnected,
    required WebBleStatusCallback onStatus,
  }) async {
    try {
      final result = await _connectBle(
        serviceUuid.toJS,
        characteristicUuid.toJS,
        ((JSString data) {
          try {
            final decoded = jsonDecode(data.toDart);

            if (decoded is List) {
              onValue(
                decoded.map((value) => value as int).toList(),
              );
            }
          } catch (error) {
            onStatus('Unable to decode BLE data: $error');
          }
        }).toJS,
        ((JSString message) {
          final status = message.toDart;

          onStatus(status);

          if (status == 'ESP32 disconnected.') {
            _connected = false;
            _deviceName = 'No Device';
            onDisconnected();
          }
        }).toJS,
      ).toDart;

      _connected = result.toDart;
      _deviceName = _bleDeviceName.toDart;

      return _connected;
    } catch (error) {
      _connected = false;
      _deviceName = 'No Device';

      onStatus('Web BLE connection failed: $error');
      return false;
    }
  }

  Future<void> disconnect() async {
    try {
      _disconnectBle();
    } catch (_) {
      // Ignore errors while disconnecting.
    }

    _connected = false;
    _deviceName = 'No Device';
  }
}
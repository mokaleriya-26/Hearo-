import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

import '../models/sound_detection.dart';
import 'notification_service.dart';

import 'web_ble_adapter_stub.dart'
    if (dart.library.html) 'web_ble_adapter_web.dart';

enum BleStatus {
  disconnected,
  connecting,
  connected,
}

enum TinyMlStatus {
  idle,
  listening,
  processing,
}

enum MicrophoneStatus {
  unavailable,
  off,
  on,
}

typedef OnSoundDetectedCallback = void Function(
  SoundDetection detection,
);

class BleService extends ChangeNotifier {
  static const String serviceUuid =
      '4fafc201-1fb5-459e-8fcc-c5c9c331914b';

  static const String characteristicUuid =
      'beb5483e-36e1-4688-b7f5-ea07361b26a8';

  BleStatus _status = BleStatus.disconnected;

  TinyMlStatus _tinyMlStatus = TinyMlStatus.idle;

  bool _isMicrophoneOn = false;
  bool _isScanning = false;

  BluetoothDevice? _connectedDevice;

  BluetoothCharacteristic? _alertCharacteristic;

  List<ScanResult> _scanResults = [];

  String _lastReceivedRawMessage = '';

  String? _statusMessage;

  SoundDetection? _currentDetection;

  StreamSubscription<List<ScanResult>>? _scanSub;

  StreamSubscription<bool>? _isScanningSub;

  StreamSubscription<BluetoothConnectionState>?
      _connectionStateSub;

  StreamSubscription<List<int>>? _charValueSub;

  Timer? _processingResetTimer;

  OnSoundDetectedCallback? onSoundDetected;

  final WebBleAdapter _webBle = WebBleAdapter();

  BleService({
    this.onSoundDetected,
  }) {
    _initBleListeners();
  }

  // --------------------------------------------------
  // GETTERS
  // --------------------------------------------------

  BleStatus get status => _status;

  TinyMlStatus get tinyMlStatus => _tinyMlStatus;

  BluetoothDevice? get connectedDevice => _connectedDevice;

  BluetoothCharacteristic? get alertCharacteristic =>
      _alertCharacteristic;

  List<ScanResult> get scanResults => _scanResults;

  bool get isScanning => _isScanning;

  bool get isConnected =>
      _status == BleStatus.connected;

  bool get isMicrophoneOn => _isMicrophoneOn;

  String get lastReceivedRawMessage =>
      _lastReceivedRawMessage;

  String? get statusMessage => _statusMessage;

  SoundDetection? get currentDetection =>
      _currentDetection;

  String get deviceName {
    if (kIsWeb) {
      return _webBle.deviceName;
    }

    final device = _connectedDevice;

    if (device == null) {
      return 'No Device';
    }

    final name = device.platformName;

    return name.isNotEmpty
        ? name
        : 'ESP32-WROOM-32 (Hearo Sensor)';
  }

  MicrophoneStatus get microphoneStatus {
    if (!isConnected) {
      return MicrophoneStatus.unavailable;
    }

    return _isMicrophoneOn
        ? MicrophoneStatus.on
        : MicrophoneStatus.off;
  }

  // --------------------------------------------------
  // INITIALIZE NATIVE BLE LISTENERS
  // --------------------------------------------------

  void _initBleListeners() {
    if (kIsWeb) {
      return;
    }

    _scanSub = FlutterBluePlus.scanResults.listen(
      (results) {
        _scanResults = results;
        notifyListeners();
      },
      onError: (Object error) {
        debugPrint('[BLE] Scan results error: $error');
      },
    );

    _isScanningSub = FlutterBluePlus.isScanning.listen(
      (scanning) {
        _isScanning = scanning;
        notifyListeners();
      },
      onError: (Object error) {
        debugPrint('[BLE] Scanning state error: $error');
      },
    );
  }

  // --------------------------------------------------
  // SCAN / CONNECT
  // --------------------------------------------------

  Future<void> startScan({
    Duration timeout = const Duration(seconds: 12),
  }) async {
    if (_isScanning) {
      return;
    }

    try {
      if (kIsWeb) {
        await _connectWebBle();
        return;
      }

      if (defaultTargetPlatform == TargetPlatform.linux) {
        _statusMessage =
            'Native BLE scanning is not configured for Linux.';
        notifyListeners();
        return;
      }

      final supported = await FlutterBluePlus.isSupported;

      if (!supported) {
        _statusMessage =
            'Bluetooth Low Energy is not supported.';
        notifyListeners();
        return;
      }

      final adapterState =
          await FlutterBluePlus.adapterState.first;

      if (adapterState != BluetoothAdapterState.on) {
        _statusMessage =
            'Please turn on Bluetooth on your device.';
        notifyListeners();
        return;
      }

      await stopScan();

      _scanResults = [];

      _statusMessage = 'Searching for ESP32...';

      notifyListeners();

      await FlutterBluePlus.startScan(
        timeout: timeout,
      );
    } catch (e) {
      _statusMessage = 'Unable to scan: $e';

      debugPrint('[BLE] Scan error: $e');

      notifyListeners();
    }
  }

  Future<void> _connectWebBle() async {
    _isScanning = true;

    _status = BleStatus.connecting;

    _isMicrophoneOn = false;

    _statusMessage =
        'Opening Chrome Bluetooth device chooser...';

    notifyListeners();

    try {
      final connected = await _webBle.connect(
        serviceUuid: serviceUuid,
        characteristicUuid: characteristicUuid,
        onValue: _onCharacteristicDataReceived,
        onDisconnected: () {
          _handleDisconnection(
            'ESP32 disconnected',
          );
        },
        onStatus: (message) {
          _statusMessage = message;
          notifyListeners();
        },
      );

      if (connected) {
        _status = BleStatus.connected;

        _isMicrophoneOn = false;

        _statusMessage =
            'Connected to ${_webBle.deviceName}. '
            'Waiting for microphone status.';

        debugPrint(
          '[BLE WEB] Connected to ${_webBle.deviceName}',
        );
      } else {
        _status = BleStatus.disconnected;

        _isMicrophoneOn = false;
      }
    } catch (e) {
      _status = BleStatus.disconnected;

      _isMicrophoneOn = false;

      _statusMessage = 'Web BLE connection failed: $e';

      debugPrint('[BLE WEB] $e');
    } finally {
      _isScanning = false;
      notifyListeners();
    }
  }

  Future<void> stopScan() async {
    try {
      if (kIsWeb) {
        _isScanning = false;
        notifyListeners();
        return;
      }

      if (FlutterBluePlus.isScanningNow) {
        await FlutterBluePlus.stopScan();
      }
    } catch (e) {
      debugPrint('[BLE] Stop scan error: $e');
    }
  }

  // --------------------------------------------------
  // NATIVE BLE CONNECTION
  // --------------------------------------------------

  Future<bool> connectToDevice(
    BluetoothDevice device,
  ) async {
    if (kIsWeb) {
      _statusMessage =
          'Use Scan Nearby BLE to select a device in Chrome.';
      notifyListeners();
      return false;
    }

    try {
      await stopScan();

      _status = BleStatus.connecting;

      _isMicrophoneOn = false;

      _statusMessage = 'Connecting to ESP32...';

      _connectedDevice = device;

      notifyListeners();

      await _charValueSub?.cancel();

      await _connectionStateSub?.cancel();

      _charValueSub = null;

      _connectionStateSub = device.connectionState.listen(
        (connectionState) {
          if (connectionState ==
              BluetoothConnectionState.connected) {
            _status = BleStatus.connected;

            _connectedDevice = device;

            _statusMessage = 'ESP32 connected';

            notifyListeners();
          } else {
            _handleDisconnection(
              'ESP32 disconnected',
            );
          }
        },
        onError: (Object error) {
          debugPrint(
            '[BLE] Connection state error: $error',
          );

          _handleDisconnection(
            'Bluetooth connection error',
          );
        },
      );

      if (device.isDisconnected) {
        await device.connect(
          license: License.nonprofit,
          timeout: const Duration(seconds: 15),
          autoConnect: false,
        );
      }

      _status = BleStatus.connected;

      _connectedDevice = device;

      notifyListeners();

      final success =
          await _discoverAndSubscribe(device);

      if (!success) {
        _statusMessage =
            'Connected, but the expected BLE characteristic '
            'was not found.';

        notifyListeners();

        return false;
      }

      _statusMessage =
          'Connected. Waiting for microphone status.';

      notifyListeners();

      final characteristic = _alertCharacteristic!;

      if (characteristic.properties.read) {
        try {
          final value = await characteristic.read();

          if (value.isNotEmpty) {
            _onCharacteristicDataReceived(value);
          }
        } catch (e) {
          debugPrint(
            '[BLE] Initial read skipped: $e',
          );
        }
      }

      return true;
    } catch (e) {
      _status = BleStatus.disconnected;

      _isMicrophoneOn = false;

      _statusMessage = 'Connection failed: $e';

      debugPrint('[BLE] Connection failed: $e');

      notifyListeners();

      return false;
    }
  }

  // --------------------------------------------------
  // DISCOVER NATIVE BLE SERVICE
  // --------------------------------------------------

  Future<bool> _discoverAndSubscribe(
    BluetoothDevice device,
  ) async {
    try {
      final services =
          await device.discoverServices();

      BluetoothCharacteristic? target;

      for (final service in services) {
        if (service.uuid.toString().toLowerCase() !=
            serviceUuid.toLowerCase()) {
          continue;
        }

        for (final characteristic
            in service.characteristics) {
          if (characteristic.uuid
                      .toString()
                      .toLowerCase() ==
                  characteristicUuid.toLowerCase() &&
              (characteristic.properties.notify ||
                  characteristic.properties.indicate)) {
            target = characteristic;
            break;
          }
        }

        if (target != null) {
          break;
        }
      }

      if (target == null) {
        debugPrint(
          '[BLE] Expected characteristic not found.',
        );

        return false;
      }

      _alertCharacteristic = target;

      await _charValueSub?.cancel();

      _charValueSub =
          target.onValueReceived.listen(
        (value) {
          if (value.isEmpty) {
            return;
          }

          debugPrint(
            '[BLE RX] ${utf8.decode(value, allowMalformed: true)}',
          );

          _onCharacteristicDataReceived(value);
        },
        onError: (Object error) {
          debugPrint(
            '[BLE] Notification error: $error',
          );
        },
      );

      await target.setNotifyValue(true);

      debugPrint('[BLE] Notifications enabled.');

      return true;
    } catch (e) {
      debugPrint('[BLE] Service discovery error: $e');

      return false;
    }
  }

  // --------------------------------------------------
  // PROCESS RECEIVED DATA
  // --------------------------------------------------

  void _onCharacteristicDataReceived(
    List<int> rawBytes,
  ) {
    try {
      final rawMessage = utf8
          .decode(rawBytes, allowMalformed: true)
          .trim();

      if (rawMessage.isEmpty) {
        return;
      }

      _lastReceivedRawMessage = rawMessage;

      debugPrint('[BLE RX] $rawMessage');

      if (rawMessage == 'MIC_ON') {
        _setMicrophoneState(true);
        return;
      }

      if (rawMessage == 'MIC_OFF') {
        _setMicrophoneState(false);
        return;
      }

      dynamic decoded;

      try {
        decoded = jsonDecode(rawMessage);
      } on FormatException {
        debugPrint('[BLE] Invalid JSON received.');

        notifyListeners();

        return;
      }

      if (decoded is! Map<String, dynamic>) {
        return;
      }

      // Expected status message:
      // {"type":"status","esp32":true,"microphone":false,"monitoring":false}
      // or {"type":"status","esp32":true,"microphone":true,"monitoring":true}
      if (decoded['type'] == 'status') {
        final bool micBool = decoded['microphone'] == true;
        final bool monBool = decoded['monitoring'] == true;
        final bool isMonitoringActive = monBool || micBool;
        _setMicrophoneState(isMonitoringActive);
        return;
      }

      // Expected detection message:
      // {"sound":"Doorbell","confidence":94.5}
      if (decoded.containsKey('sound') &&
          decoded.containsKey('confidence')) {
        try {
          final detection = SoundDetection.fromBleJson(
            decoded,
            deviceId: deviceName,
          );

          _currentDetection = detection;

          _tinyMlStatus = TinyMlStatus.processing;

          _processingResetTimer?.cancel();

          _processingResetTimer = Timer(
            const Duration(seconds: 2),
            () {
              if (isConnected) {
                _tinyMlStatus = _isMicrophoneOn
                    ? TinyMlStatus.listening
                    : TinyMlStatus.idle;

                notifyListeners();
              }
            },
          );

          _triggerAccessibilityFeedback(detection);

          // Trigger local safety notification for dangerous sound events (with deduplication)
          NotificationService().handleDangerousSoundAlert(detection);

          onSoundDetected?.call(detection);

          notifyListeners();
        } catch (e) {
          debugPrint(
            '[BLE] Detection parsing error: $e',
          );
        }
      }
    } catch (e) {
      debugPrint(
        '[BLE] Message processing error: $e',
      );
    }
  }

  // --------------------------------------------------
  // UPDATE MICROPHONE STATUS
  // --------------------------------------------------

  void _setMicrophoneState(bool isOn) {
    _isMicrophoneOn = isOn;

    _tinyMlStatus = isOn
        ? TinyMlStatus.listening
        : TinyMlStatus.idle;

    _statusMessage = isOn
        ? 'Microphone ON — Listening for sounds'
        : 'Microphone OFF';

    debugPrint(
      '[BLE] Microphone is ${isOn ? "ON" : "OFF"}',
    );

    if (!isOn) {
      NotificationService().resetEventState();
    }

    notifyListeners();
  }

  // --------------------------------------------------
  // ACCESSIBILITY FEEDBACK
  // --------------------------------------------------

  void _triggerAccessibilityFeedback(
    SoundDetection detection,
  ) {
    try {
      final category = detection.category;

      if (!category.triggersVibration) {
        return;
      }

      if (category.isCritical) {
        HapticFeedback.heavyImpact();

        Future.delayed(
          const Duration(milliseconds: 250),
          () {
            if (!kIsWeb) {
              HapticFeedback.heavyImpact();
            }
          },
        );
      } else {
        HapticFeedback.mediumImpact();
      }
    } catch (e) {
      debugPrint('[BLE] Haptic feedback error: $e');
    }
  }

  // --------------------------------------------------
  // DISCONNECT
  // --------------------------------------------------

  Future<void> disconnect() async {
    try {
      if (kIsWeb) {
        await _webBle.disconnect();

        _handleDisconnection(
          'ESP32 disconnected',
        );

        return;
      }

      await _charValueSub?.cancel();

      _charValueSub = null;

      await _connectionStateSub?.cancel();

      _connectionStateSub = null;

      final device = _connectedDevice;

      if (device != null && device.isConnected) {
        await device.disconnect();
      }
    } catch (e) {
      debugPrint('[BLE] Disconnect error: $e');
    } finally {
      _handleDisconnection(
        'ESP32 disconnected',
      );
    }
  }

  void _handleDisconnection(String message) {
    _processingResetTimer?.cancel();

    _status = BleStatus.disconnected;

    _tinyMlStatus = TinyMlStatus.idle;

    _isMicrophoneOn = false;

    _alertCharacteristic = null;

    _connectedDevice = null;

    _statusMessage = message;

    NotificationService().resetEventState();

    debugPrint('[BLE] $message');

    notifyListeners();
  }

  // --------------------------------------------------
  // TESTING HELPER
  // --------------------------------------------------

  void simulateBlePacketForTesting({
    required String sound,
    required double confidence,
  }) {
    final message = jsonEncode({
      'sound': sound,
      'confidence': confidence,
      'timestamp': DateTime.now().toIso8601String(),
    });

    _onCharacteristicDataReceived(
      utf8.encode(message),
    );
  }

  // --------------------------------------------------
  // CLEANUP
  // --------------------------------------------------

  @override
  void dispose() {
    _scanSub?.cancel();

    _isScanningSub?.cancel();

    _connectionStateSub?.cancel();

    _charValueSub?.cancel();

    _processingResetTimer?.cancel();

    if (kIsWeb) {
      _webBle.disconnect();
    }

    super.dispose();
  }
}
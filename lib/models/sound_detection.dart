import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';
import 'sound_category.dart';

class SoundDetection {
  final String? id;
  final String sound;
  final double confidence;
  final DateTime timestamp;
  final String deviceId;
  final String userId;

  SoundDetection({
    this.id,
    required this.sound,
    required this.confidence,
    required this.timestamp,
    this.deviceId = 'ESP32-WROOM-32',
    this.userId = '',
  });

  /// Factory constructor to parse incoming BLE JSON packet:
  /// e.g. {"sound": "Doorbell", "confidence": 94.5, "timestamp": "2026-10-07T14:35:20"}
  /// or {"sound": "Doorbell", "confidence": 94.5}
  factory SoundDetection.fromBleJson(
    Map<String, dynamic> json, {
    String deviceId = 'ESP32-WROOM-32',
    String userId = '',
  }) {
    DateTime parsedTime = DateTime.now();
    if (json.containsKey('timestamp') && json['timestamp'] != null) {
      try {
        parsedTime = DateTime.parse(json['timestamp'].toString());
      } catch (_) {
        parsedTime = DateTime.now();
      }
    }

    final double parsedConfidence = double.tryParse(
          json['confidence']?.toString() ?? '0',
        ) ??
        0.0;

    return SoundDetection(
      sound: json['sound']?.toString() ?? 'Unknown Sound',
      confidence: parsedConfidence > 1.0 && parsedConfidence <= 100.0
          ? parsedConfidence
          : (parsedConfidence <= 1.0 ? parsedConfidence * 100.0 : parsedConfidence),
      timestamp: parsedTime,
      deviceId: json['deviceId']?.toString() ?? deviceId,
      userId: userId,
    );
  }

  /// Factory constructor to parse Firestore DocumentSnapshot
  factory SoundDetection.fromFirestore(DocumentSnapshot<Map<String, dynamic>> doc) {
    final data = doc.data() ?? {};
    DateTime parsedTime = DateTime.now();

    final rawTimestamp = data['timestamp'];
    if (rawTimestamp is Timestamp) {
      parsedTime = rawTimestamp.toDate();
    } else if (rawTimestamp is String) {
      parsedTime = DateTime.tryParse(rawTimestamp) ?? DateTime.now();
    }

    final double conf = double.tryParse(data['confidence']?.toString() ?? '0') ?? 0.0;

    return SoundDetection(
      id: doc.id,
      sound: data['sound']?.toString() ?? 'Unknown Sound',
      confidence: conf > 1.0 && conf <= 100.0 ? conf : (conf <= 1.0 ? conf * 100.0 : conf),
      timestamp: parsedTime,
      deviceId: data['deviceId']?.toString() ?? 'ESP32-WROOM-32',
      userId: data['userId']?.toString() ?? '',
    );
  }

  /// Convert to Firestore Map with server timestamp
  Map<String, dynamic> toFirestoreMap({bool useServerTimestamp = true}) {
    return {
      'sound': sound,
      'confidence': double.parse(confidence.toStringAsFixed(1)),
      'timestamp': useServerTimestamp ? FieldValue.serverTimestamp() : Timestamp.fromDate(timestamp),
      'deviceId': deviceId,
      'userId': userId,
    };
  }

  /// Convert to JSON Map
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'sound': sound,
      'confidence': double.parse(confidence.toStringAsFixed(1)),
      'timestamp': timestamp.toIso8601String(),
      'deviceId': deviceId,
      'userId': userId,
    };
  }

  /// Get the category configuration for visual styling and alerts
  SoundCategory get category => SoundCategoryRegistry.getCategoryByName(sound);

  /// Formatted date string: "07 Oct 2026"
  String get formattedDate => DateFormat('dd MMM yyyy').format(timestamp);

  /// Formatted time string: "02:35 PM"
  String get formattedTime => DateFormat('hh:mm a').format(timestamp);

  /// Confidence display string: "94.5%" or "92%"
  String get confidenceDisplay {
    if (confidence % 1 == 0) {
      return '${confidence.toInt()}%';
    }
    return '${confidence.toStringAsFixed(1)}%';
  }

  /// Formatted display title (e.g. "Bell Sounds" instead of "bell_sounds")
  String get displayTitle => category.displayName.isNotEmpty ? category.displayName : sound;

  /// Clean display line: "Doorbell | 94% | 07 Oct 2026 | 02:35 PM"
  String get summaryLine => '$displayTitle | ${confidence.round()}% | $formattedDate | $formattedTime';
}

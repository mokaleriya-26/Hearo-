import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/sound_category.dart';
import '../models/sound_detection.dart';

/// Dedicated service responsible for local device notifications in Hearo.
/// Handles high-priority alerts for DANGEROUS sound events with deduplication
/// and rate-limiting to prevent notification spam.
class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  bool _isInitialized = false;
  bool _notificationsEnabled = true;

  // Deduplication & spam prevention state
  String? _lastDangerousClass;
  DateTime? _lastDangerousAlertTime;
  Duration cooldownDuration = const Duration(seconds: 10);

  // Channel constants
  static const String channelId = 'hearo_safety_alerts';
  static const String channelName = 'Hearo Safety Alerts';
  static const String channelDescription =
      'Urgent local safety alerts when dangerous sounds like sirens, alarms, and car horns are detected.';

  bool get isInitialized => _isInitialized;
  bool get notificationsEnabled => _notificationsEnabled;

  set notificationsEnabled(bool value) {
    _notificationsEnabled = value;
  }

  /// Initialize local notification plugins and high-priority channel
  Future<void> initialize({
    void Function(NotificationResponse)? onNotificationTapped,
  }) async {
    if (_isInitialized) {
      return;
    }

    try {
      const AndroidInitializationSettings androidSettings =
          AndroidInitializationSettings('@mipmap/ic_launcher');

      const DarwinInitializationSettings iosSettings =
          DarwinInitializationSettings(
        requestAlertPermission: false,
        requestBadgePermission: false,
        requestSoundPermission: false,
      );

      const InitializationSettings initSettings = InitializationSettings(
        android: androidSettings,
        iOS: iosSettings,
      );

      await _notificationsPlugin.initialize(
        settings: initSettings,
        onDidReceiveNotificationResponse: onNotificationTapped,
      );

      // Create Android Notification Channel with high importance and vibration
      final androidImplementation = _notificationsPlugin
          .resolvePlatformSpecificImplementation<
              AndroidFlutterLocalNotificationsPlugin>();

      if (androidImplementation != null) {
        final vibrationPattern = Int64List.fromList([0, 500, 200, 500]);
        final androidChannel = AndroidNotificationChannel(
          channelId,
          channelName,
          description: channelDescription,
          importance: Importance.max,
          enableVibration: true,
          vibrationPattern: vibrationPattern,
          playSound: true,
        );

        await androidImplementation.createNotificationChannel(androidChannel);
      }

      _isInitialized = true;
      debugPrint('[Hearo Notifications] Initialized successfully.');
    } catch (e) {
      debugPrint('[Hearo Notifications] Initialization error: $e');
    }
  }

  /// Request runtime notification permission (Android 13+ and iOS)
  Future<bool> requestPermissions() async {
    try {
      if (kIsWeb) {
        return false;
      }

      if (Platform.isAndroid) {
        final androidImplementation = _notificationsPlugin
            .resolvePlatformSpecificImplementation<
                AndroidFlutterLocalNotificationsPlugin>();

        final granted =
            await androidImplementation?.requestNotificationsPermission();

        if (granted == true) {
          return true;
        }

        // Fallback check with permission_handler
        final status = await Permission.notification.request();
        return status.isGranted;
      } else if (Platform.isIOS) {
        final iosImplementation = _notificationsPlugin
            .resolvePlatformSpecificImplementation<
                IOSFlutterLocalNotificationsPlugin>();

        final granted = await iosImplementation?.requestPermissions(
          alert: true,
          badge: true,
          sound: true,
        );
        return granted ?? false;
      }
      return false;
    } catch (e) {
      debugPrint('[Hearo Notifications] Request permission error: $e');
      return false;
    }
  }

  /// Check whether notification permission is currently granted
  Future<bool> isPermissionGranted() async {
    try {
      if (kIsWeb) return false;
      if (Platform.isAndroid || Platform.isIOS) {
        return await Permission.notification.isGranted;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Handles incoming sound predictions from ESP32.
  /// If classified as DANGEROUS, verifies deduplication cooldown and triggers notification.
  Future<bool> handleDangerousSoundAlert(SoundDetection detection) async {
    if (!_notificationsEnabled) {
      return false;
    }

    final category = detection.category;

    // Only DANGEROUS sound categories trigger local notifications
    if (category.riskLevel != SoundRiskLevel.dangerous) {
      // If a non-dangerous sound is received, reset active event after a brief delay
      if (_lastDangerousClass != null) {
        _lastDangerousClass = null;
      }
      return false;
    }

    // Check deduplication and cooldown
    if (!shouldTriggerNotification(detection)) {
      debugPrint(
        '[Hearo Notifications] Throttling repeated alert for "${detection.displayTitle}" (within cooldown).',
      );
      return false;
    }

    return await _showDangerousNotification(detection);
  }

  /// Check whether this dangerous detection should generate a notification
  bool shouldTriggerNotification(SoundDetection detection, [DateTime? testNow]) {
    final currentClass = detection.category.id;
    final now = testNow ?? DateTime.now();

    // 1. If it's a DIFFERENT dangerous sound class, notify immediately!
    if (_lastDangerousClass != currentClass) {
      _lastDangerousClass = currentClass;
      _lastDangerousAlertTime = now;
      return true;
    }

    // 2. If it's the SAME dangerous class, enforce cooldown
    if (_lastDangerousAlertTime == null ||
        now.difference(_lastDangerousAlertTime!) >= cooldownDuration) {
      _lastDangerousAlertTime = now;
      return true;
    }

    // Still inside cooldown for the same dangerous class
    return false;
  }

  /// Formats the notification title as required
  String formatNotificationTitle(SoundDetection detection) {
    return '⚠️ Hearo Safety Alert';
  }

  /// Formats the notification body adapted to detected class and confidence
  String formatNotificationBody(SoundDetection detection) {
    final soundName = detection.displayTitle;
    return detection.confidence > 0
        ? 'Dangerous sound detected: $soundName (${detection.confidenceDisplay}). Please check your surroundings.'
        : 'Dangerous sound detected: $soundName. Please check your surroundings.';
  }

  /// Reset deduplication tracking (e.g. when monitoring stops)
  void resetEventState() {
    _lastDangerousClass = null;
    _lastDangerousAlertTime = null;
  }

  /// Build and display the immediate local notification
  Future<bool> _showDangerousNotification(SoundDetection detection) async {
    try {
      final category = detection.category;
      final String title = formatNotificationTitle(detection);
      final String body = formatNotificationBody(detection);

      final vibrationPattern = Int64List.fromList([0, 500, 200, 500]);

      final AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
        channelId,
        channelName,
        channelDescription: channelDescription,
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'Hearo Dangerous Sound Alert',
        icon: '@mipmap/ic_launcher',
        vibrationPattern: vibrationPattern,
        enableVibration: true,
        playSound: true,
        color: category.primaryColor,
        styleInformation: BigTextStyleInformation(
          body,
          contentTitle: title,
          summaryText: 'Hearo Emergency Alert',
        ),
      );

      final DarwinNotificationDetails iosDetails = DarwinNotificationDetails(
        presentAlert: true,
        presentBadge: true,
        presentSound: true,
        interruptionLevel: InterruptionLevel.critical,
      );

      final NotificationDetails notificationDetails = NotificationDetails(
        android: androidDetails,
        iOS: iosDetails,
      );

      // Distinct notification ID for each dangerous sound class so they update cleanly
      final int notificationId = category.id.hashCode & 0x7FFFFFFF;

      await _notificationsPlugin.show(
        id: notificationId,
        title: title,
        body: body,
        notificationDetails: notificationDetails,
        payload: detection.sound,
      );

      debugPrint(
        '[Hearo Notifications] Dispatched dangerous alert notification: $title - $body',
      );
      return true;
    } catch (e) {
      debugPrint('[Hearo Notifications] Error displaying notification: $e');
      return false;
    }
  }

  /// Cancel all active notifications
  Future<void> cancelAll() async {
    try {
      await _notificationsPlugin.cancelAll();
    } catch (_) {}
  }
}

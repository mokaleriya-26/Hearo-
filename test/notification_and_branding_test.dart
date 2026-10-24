import 'package:flutter_test/flutter_test.dart';
import 'package:sound_alert_app/models/sound_category.dart';
import 'package:sound_alert_app/models/sound_detection.dart';
import 'package:sound_alert_app/services/notification_service.dart';

void main() {
  group('Hearo NotificationService & Risk Deduplication Tests', () {
    late NotificationService notificationService;

    setUp(() {
      notificationService = NotificationService();
      notificationService.notificationsEnabled = true;
      notificationService.resetEventState();
      notificationService.cooldownDuration = const Duration(seconds: 10);
    });

    test('Risk classification: Only DANGEROUS sound categories trigger notifications', () async {
      // 6 Dangerous Classes: car_horn, explosive_sounds, general_alarm, glass_breaking, screaming, siren
      final dangerousClasses = [
        'car_horn',
        'explosive_sounds',
        'general_alarm',
        'glass_breaking',
        'screaming',
        'siren',
      ];

      for (final id in dangerousClasses) {
        final detection = SoundDetection(
          sound: id,
          confidence: 90.0,
          timestamp: DateTime.now(),
          deviceId: 'ESP32-WROOM-32',
        );
        expect(detection.category.riskLevel, equals(SoundRiskLevel.dangerous));
        expect(detection.category.isCritical, isTrue);
      }

      // Medium and Safe classes must NOT be classified as dangerous
      final nonDangerousClasses = [
        'bell_sounds',
        'cat',
        'cheering',
        'clapping',
        'clock',
        'coughing',
        'crying_sobbing',
        'dog',
        'door_knock',
        'doorbell',
        'footsteps',
        'laughter',
        'phone_ringtone',
        'transport_sounds',
      ];

      for (final id in nonDangerousClasses) {
        final detection = SoundDetection(
          sound: id,
          confidence: 99.0, // High confidence safe sound
          timestamp: DateTime.now(),
          deviceId: 'ESP32-WROOM-32',
        );
        expect(detection.category.riskLevel, isNot(equals(SoundRiskLevel.dangerous)));

        // Attempting to handle a safe/medium sound must return false and NOT trigger notification
        final triggered = await notificationService.handleDangerousSoundAlert(detection);
        expect(triggered, isFalse, reason: '$id should not trigger a safety notification');
      }
    });

    test('Unknown sound class does not default to dangerous', () async {
      final unknownDetection = SoundDetection(
        sound: 'mysterious_alien_sound',
        confidence: 99.0,
        timestamp: DateTime.now(),
        deviceId: 'ESP32-WROOM-32',
      );

      expect(unknownDetection.category.riskLevel, isNot(equals(SoundRiskLevel.dangerous)));
      final triggered = await notificationService.handleDangerousSoundAlert(unknownDetection);
      expect(triggered, isFalse);
    });

    test('Notification formatting matches exact requirements', () {
      final detection = SoundDetection(
        sound: 'glass_breaking',
        confidence: 92.0,
        timestamp: DateTime.now(),
        deviceId: 'ESP32-WROOM-32',
      );

      final title = notificationService.formatNotificationTitle(detection);
      final body = notificationService.formatNotificationBody(detection);

      // Title must be: ⚠️ Hearo Safety Alert
      expect(title, equals('⚠️ Hearo Safety Alert'));

      // Body must be: Dangerous sound detected: Glass Breaking (92%). Please check your surroundings.
      expect(body, contains('Dangerous sound detected: Glass Breaking (92%)'));
      expect(body, contains('Please check your surroundings.'));

      // Siren test
      final sirenDetection = SoundDetection(
        sound: 'siren',
        confidence: 95.4,
        timestamp: DateTime.now(),
        deviceId: 'ESP32-WROOM-32',
      );
      final sirenBody = notificationService.formatNotificationBody(sirenDetection);
      expect(sirenBody, equals('Dangerous sound detected: Siren (95.4%). Please check your surroundings.'));
    });

    test('Deduplication: Same dangerous sound throttled within 10s cooldown', () {
      final baseTime = DateTime(2026, 10, 24, 12, 0, 0);

      final siren1 = SoundDetection(
        sound: 'siren',
        confidence: 88.0,
        timestamp: baseTime,
        deviceId: 'ESP32-WROOM-32',
      );

      // 1st prediction at t=0s: Must trigger
      final firstTrigger = notificationService.shouldTriggerNotification(siren1, baseTime);
      expect(firstTrigger, isTrue);

      // 2nd prediction at t=3s (same sound): Must be throttled / deduplicated
      final secondTrigger = notificationService.shouldTriggerNotification(
        siren1,
        baseTime.add(const Duration(seconds: 3)),
      );
      expect(secondTrigger, isFalse);

      // 3rd prediction at t=9s (same sound): Still within 10s cooldown, must be throttled
      final thirdTrigger = notificationService.shouldTriggerNotification(
        siren1,
        baseTime.add(const Duration(seconds: 9)),
      );
      expect(thirdTrigger, isFalse);

      // 4th prediction at t=11s (>10s): Cooldown expired, must trigger
      final fourthTrigger = notificationService.shouldTriggerNotification(
        siren1,
        baseTime.add(const Duration(seconds: 11)),
      );
      expect(fourthTrigger, isTrue);
    });

    test('Deduplication: Different dangerous sound triggers immediately without waiting', () {
      final baseTime = DateTime(2026, 10, 24, 12, 0, 0);

      final siren = SoundDetection(
        sound: 'siren',
        confidence: 90.0,
        timestamp: baseTime,
        deviceId: 'ESP32-WROOM-32',
      );

      final carHorn = SoundDetection(
        sound: 'car_horn',
        confidence: 85.0,
        timestamp: baseTime.add(const Duration(seconds: 2)),
        deviceId: 'ESP32-WROOM-32',
      );

      // 1. Siren at t=0s: Triggers
      expect(notificationService.shouldTriggerNotification(siren, baseTime), isTrue);

      // 2. Car horn at t=2s: DIFFERENT dangerous class, must trigger immediately!
      expect(
        notificationService.shouldTriggerNotification(carHorn, baseTime.add(const Duration(seconds: 2))),
        isTrue,
      );

      // 3. Repeated car horn at t=4s: Same as last class, throttled!
      expect(
        notificationService.shouldTriggerNotification(carHorn, baseTime.add(const Duration(seconds: 4))),
        isFalse,
      );
    });

    test('Resetting event state clears deduplication', () {
      final baseTime = DateTime(2026, 10, 24, 12, 0, 0);

      final siren = SoundDetection(
        sound: 'siren',
        confidence: 90.0,
        timestamp: baseTime,
        deviceId: 'ESP32-WROOM-32',
      );

      expect(notificationService.shouldTriggerNotification(siren, baseTime), isTrue);

      // Reset state (e.g. disconnect or mic off)
      notificationService.resetEventState();

      // Next prediction for siren at t=2s must trigger again because state was reset
      expect(
        notificationService.shouldTriggerNotification(siren, baseTime.add(const Duration(seconds: 2))),
        isTrue,
      );
    });

    test('When notificationsEnabled is false, notifications are suppressed', () async {
      notificationService.notificationsEnabled = false;

      final siren = SoundDetection(
        sound: 'siren',
        confidence: 99.0,
        timestamp: DateTime.now(),
        deviceId: 'ESP32-WROOM-32',
      );

      final triggered = await notificationService.handleDangerousSoundAlert(siren);
      expect(triggered, isFalse);
    });
  });
}

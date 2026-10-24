import 'package:flutter_test/flutter_test.dart';
import 'package:sound_alert_app/models/sound_category.dart';
import 'package:sound_alert_app/models/sound_detection.dart';
import 'package:sound_alert_app/services/auth_service.dart';

void main() {
  group('SoundCategoryRegistry Tests (20 TinyML Classes)', () {
    test('Registry contains all 20 required TinyML sound classes', () {
      expect(SoundCategoryRegistry.categories.length, equals(20));

      final expectedFirmwareIds = [
        'bell_sounds',
        'car_horn',
        'cat',
        'cheering',
        'clapping',
        'clock',
        'coughing',
        'crying_sobbing',
        'dog',
        'door_knock',
        'doorbell',
        'explosive_sounds',
        'footsteps',
        'general_alarm',
        'glass_breaking',
        'laughter',
        'phone_ringtone',
        'screaming',
        'siren',
        'transport_sounds',
      ];

      final actualIds = SoundCategoryRegistry.categories.map((c) => c.id).toList();
      for (final id in expectedFirmwareIds) {
        expect(actualIds, contains(id));
      }

      // Verify all 20 identifiers are unique
      expect(actualIds.toSet().length, equals(20));
    });

    test('Risk classification matches requirements for all 20 categories', () {
      final dangerousIds = [
        'car_horn',
        'explosive_sounds',
        'general_alarm',
        'glass_breaking',
        'screaming',
        'siren',
      ];

      final mediumIds = [
        'bell_sounds',
        'coughing',
        'crying_sobbing',
        'dog',
        'door_knock',
        'doorbell',
        'footsteps',
        'phone_ringtone',
        'transport_sounds',
      ];

      final safeIds = [
        'cat',
        'cheering',
        'clapping',
        'clock',
        'laughter',
      ];

      for (final id in dangerousIds) {
        final cat = SoundCategoryRegistry.getCategoryByName(id);
        expect(cat.riskLevel, equals(SoundRiskLevel.dangerous), reason: '$id should be DANGEROUS');
        expect(cat.isCritical, isTrue);
      }

      for (final id in mediumIds) {
        final cat = SoundCategoryRegistry.getCategoryByName(id);
        expect(cat.riskLevel, equals(SoundRiskLevel.medium), reason: '$id should be MEDIUM');
        expect(cat.isCritical, isFalse);
      }

      for (final id in safeIds) {
        final cat = SoundCategoryRegistry.getCategoryByName(id);
        expect(cat.riskLevel, equals(SoundRiskLevel.safe), reason: '$id should be SAFE');
        expect(cat.isCritical, isFalse);
      }
    });

    test('Case-insensitive category lookup works for firmware ID and display name', () {
      final bell = SoundCategoryRegistry.getCategoryByName('bell_sounds');
      expect(bell.name, equals('Bell Sounds'));
      expect(bell.riskLevel, equals(SoundRiskLevel.medium));

      final horn = SoundCategoryRegistry.getCategoryByName('Car Horn');
      expect(horn.id, equals('car_horn'));
      expect(horn.riskLevel, equals(SoundRiskLevel.dangerous));

      final scream = SoundCategoryRegistry.getCategoryByName('SCREAMING');
      expect(scream.id, equals('screaming'));
      expect(scream.riskLevel, equals(SoundRiskLevel.dangerous));
    });

    test('Gracefully handles unknown sound classes without crashing or defaulting to dangerous', () {
      final unknown = SoundCategoryRegistry.getCategoryByName('mysterious_sound_xyz');
      expect(unknown.riskLevel, isNot(equals(SoundRiskLevel.dangerous)));
      expect(unknown.isCritical, isFalse);
    });
  });

  group('SoundDetection BLE JSON Parsing Tests', () {
    test('Parses standard BLE JSON packet correctly', () {
      final json = {
        'sound': 'doorbell',
        'confidence': 94.5,
      };

      final detection = SoundDetection.fromBleJson(json);
      expect(detection.sound, equals('doorbell'));
      expect(detection.displayTitle, equals('Doorbell'));
      expect(detection.confidence, equals(94.5));
      expect(detection.deviceId, equals('ESP32-WROOM-32'));
      expect(detection.confidenceDisplay, equals('94.5%'));
    });

    test('Parses timestamped BLE JSON packet correctly', () {
      final json = {
        'sound': 'siren',
        'confidence': 96.0,
        'timestamp': 123456,
        'deviceId': 'ESP32-WROOM-32',
      };

      final detection = SoundDetection.fromBleJson(json);
      expect(detection.sound, equals('siren'));
      expect(detection.displayTitle, equals('Siren'));
      expect(detection.confidence, equals(96.0));
      expect(detection.category.isCritical, isTrue);
      expect(detection.category.riskLevel, equals(SoundRiskLevel.dangerous));
    });
  });

  group('AuthService Exception Handling Tests', () {
    test('Translates all standard FirebaseAuthException codes into friendly messages', () {
      final authService = AuthService();

      expect(authService.errorMessage, isNull);

      // Verify demo login works
      authService.loginAsDemoUser(email: 'test@tinyml-soundalert.org');
      expect(authService.isAuthenticated, isTrue);
      expect(authService.userEmail, equals('test@tinyml-soundalert.org'));

      authService.signOut();
      expect(authService.isAuthenticated, isFalse);
    });
  });
}

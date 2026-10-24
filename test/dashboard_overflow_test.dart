import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sound_alert_app/models/sound_detection.dart';
import 'package:sound_alert_app/theme/app_theme.dart';
import 'package:sound_alert_app/widgets/sound_alert_card.dart';

void main() {
  group('Dashboard SoundAlertCard Responsive Layout & Overflow Tests', () {
    testWidgets('Renders properly without overflow on ultra-narrow 320px screen', (tester) async {
      // Set small physical screen size (width 320px)
      tester.view.physicalSize = const Size(320, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final detection = SoundDetection(
        sound: 'explosive_sounds',
        confidence: 98.7,
        timestamp: DateTime(2026, 10, 24, 19, 35),
        deviceId: 'ESP32-WROOM-32',
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: SoundAlertCard(
                  detection: detection,
                  isConnected: true,
                ),
              ),
            ),
          ),
        ),
      );

      await tester.pump();

      expect(tester.takeException(), isNull);
      expect(find.text('EXPLOSIVE SOUNDS DETECTED'), findsOneWidget);
      expect(find.text('98.7%'), findsOneWidget);
      expect(find.text('DANGEROUS'), findsOneWidget);
      expect(find.text('07:35 PM'), findsOneWidget);
    });

    testWidgets('Renders long category name (transport_sounds) on 360px screen without overflow', (tester) async {
      tester.view.physicalSize = const Size(360, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final detection = SoundDetection(
        sound: 'transport_sounds',
        confidence: 84.2,
        timestamp: DateTime(2026, 10, 24, 14, 5),
        deviceId: 'ESP32-WROOM-32',
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: SoundAlertCard(
                  detection: detection,
                  isConnected: true,
                ),
              ),
            ),
          ),
        ),
      );

      await tester.pump();

      expect(tester.takeException(), isNull);
      expect(find.text('TRANSPORT SOUNDS DETECTED'), findsOneWidget);
      expect(find.text('MEDIUM'), findsOneWidget);
      expect(find.text('02:05 PM'), findsOneWidget);
    });

    testWidgets('Renders disconnected state with ESP32-WROOM-32 description without overflow', (tester) async {
      tester.view.physicalSize = const Size(320, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const Scaffold(
            body: Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: SoundAlertCard(
                detection: null,
                isConnected: false,
              ),
            ),
          ),
        ),
      );

      await tester.pump();

      expect(tester.takeException(), isNull);
      expect(find.text('ESP32 Not Connected'), findsOneWidget);
      expect(find.textContaining('ESP32-WROOM-32'), findsOneWidget);
    });
  });
}

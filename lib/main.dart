import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:provider/provider.dart';

import 'firebase_options.dart';
import 'theme/app_theme.dart';
import 'services/auth_service.dart';
import 'services/firebase_service.dart';
import 'services/ble_service.dart';
import 'screens/splash_screen.dart';

import 'services/notification_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Set system UI overlay style for dark high-contrast immersion
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: AppTheme.background,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  // Initialize Local Notifications Service during startup
  try {
    await NotificationService().initialize();
  } catch (e) {
    debugPrint('[Hearo] NotificationService init note: $e');
  }

  // Initialize Firebase (safely handles initial missing credentials)
  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
    debugPrint('Firebase Core initialized successfully.');
  } catch (e) {
    debugPrint('Firebase initialization note: $e');
  }

  runApp(const HearoApp());
}

/// Main entry widget for Hearo assistive sound detection
typedef TinyMlSoundAlertApp = HearoApp;

class HearoApp extends StatelessWidget {
  const HearoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider<AuthService>(
          create: (_) => AuthService(),
        ),
        ChangeNotifierProvider<FirestoreAlertService>(
          create: (_) => FirestoreAlertService(),
        ),
        ChangeNotifierProxyProvider2<AuthService, FirestoreAlertService, BleService>(
          create: (context) {
            final auth = Provider.of<AuthService>(context, listen: false);
            final firebase = Provider.of<FirestoreAlertService>(context, listen: false);

            final bleService = BleService();
            bleService.onSoundDetected = (detection) {
              // Automatically persist valid detection event to Cloud Firestore
              firebase.saveDetection(detection, userId: auth.userId);
            };
            return bleService;
          },
          update: (context, auth, firebase, previousBle) {
            final ble = previousBle ?? BleService();
            ble.onSoundDetected = (detection) {
              firebase.saveDetection(detection, userId: auth.userId);
            };
            return ble;
          },
        ),
      ],
      child: MaterialApp(
        title: 'Hearo - Every Sound Matters.',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.darkTheme,
        home: const SplashScreen(),
      ),
    );
  }
}

// File generated for Flutter Firebase setup.
// To connect to your real Firebase project:
// Run: `flutterfire configure` or replace the values below with your Firebase Project configurations.

import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        return windows;
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for linux - '
          'you can reconfigure this by running the FlutterFire CLI.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyDrL2C_hMaHIU3XXTswALJNSreLeEpyy78',
    appId: '1:949352814912:web:05aa77bb5cb032a9f782a5',
    messagingSenderId: '949352814912',
    projectId: 'smart-sound-alert',
    authDomain: 'smart-sound-alert.firebaseapp.com',
    databaseURL: 'https://smart-sound-alert-default-rtdb.firebaseio.com',
    storageBucket: 'smart-sound-alert.firebasestorage.app',
    measurementId: 'G-PWV16LTRE9',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyDEth5W-vzXu0s3kACOT_ofiD-v1oty-Hw',
    appId: '1:949352814912:android:0de3fdec4ed20b5ff782a5',
    messagingSenderId: '949352814912',
    projectId: 'smart-sound-alert',
    databaseURL: 'https://smart-sound-alert-default-rtdb.firebaseio.com',
    storageBucket: 'smart-sound-alert.firebasestorage.app',
  );
  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyAMSUy8_7mjzNrnCjIKmAjcC15v6-CFVKA',
    appId: '1:949352814912:ios:eb89add382aee6e8f782a5',
    messagingSenderId: '949352814912',
    projectId: 'smart-sound-alert',
    databaseURL: 'https://smart-sound-alert-default-rtdb.firebaseio.com',
    storageBucket: 'smart-sound-alert.firebasestorage.app',
    iosBundleId: 'com.tinyml.soundalert.soundAlertApp',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyAMSUy8_7mjzNrnCjIKmAjcC15v6-CFVKA',
    appId: '1:949352814912:ios:eb89add382aee6e8f782a5',
    messagingSenderId: '949352814912',
    projectId: 'smart-sound-alert',
    databaseURL: 'https://smart-sound-alert-default-rtdb.firebaseio.com',
    storageBucket: 'smart-sound-alert.firebasestorage.app',
    iosBundleId: 'com.tinyml.soundalert.soundAlertApp',
  );

  static const FirebaseOptions windows = FirebaseOptions(
    apiKey: 'AIzaSyDemoDummyKeyForFlutterCompilation12345678',
    appId: '1:1234567890:web:abcdef1234567890',
    messagingSenderId: '1234567890',
    projectId: 'tinyml-sound-alert',
    storageBucket: 'tinyml-sound-alert.appspot.com',
  );
}

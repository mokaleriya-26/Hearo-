import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';

import 'package:firebase_core/firebase_core.dart';

class AuthService extends ChangeNotifier {
  final FirebaseAuth? _authOverride;
  FirebaseAuth get _auth => _authOverride ?? FirebaseAuth.instance;

  User? _currentUser;
  bool _isLoading = false;
  String? _errorMessage;

  // Demo user mode for offline presentation/testing
  bool _isDemoMode = false;
  String? _demoEmail;

  AuthService({FirebaseAuth? auth}) : _authOverride = auth {
    _init();
  }

  void _init() {
    try {
      if (Firebase.apps.isNotEmpty || _authOverride != null) {
        _auth.authStateChanges().listen((User? user) {
          if (!_isDemoMode) {
            _currentUser = user;
            notifyListeners();
          }
        });
      }
    } catch (e) {
      debugPrint('[AuthService] init note: $e');
    }
  }

  User? get currentUser => _currentUser;
  bool get isAuthenticated => _currentUser != null || _isDemoMode;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  bool get isDemoMode => _isDemoMode;

  String get userId {
    if (_currentUser != null) return _currentUser!.uid;
    if (_isDemoMode) return 'demo_user_esp32_deaf_alert';
    return 'anonymous_user';
  }

  String get userEmail {
    if (_currentUser != null && _currentUser!.email != null) {
      return _currentUser!.email!;
    }
    if (_isDemoMode && _demoEmail != null) {
      return _demoEmail!;
    }
    return 'Guest User';
  }

  Stream<User?> get authStateChanges {
    try {
      if (Firebase.apps.isNotEmpty || _authOverride != null) {
        return _auth.authStateChanges();
      }
    } catch (_) {}
    return const Stream.empty();
  }

  /// Sign in with email and password
  Future<bool> signInWithEmailPassword(String email, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      debugPrint('[AuthService] Attempting signIn for: ${email.trim()}');
      final credential = await _auth.signInWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
      _currentUser = credential.user;
      _isDemoMode = false;
      _isLoading = false;
      _errorMessage = null;
      notifyListeners();
      debugPrint('[AuthService] signIn successful: UID=${_currentUser?.uid}');
      return true;
    } on FirebaseAuthException catch (e) {
      debugPrint('[AuthService] FirebaseAuthException during signIn: code=${e.code}, message=${e.message}');
      _errorMessage = _getReadableAuthError(e.code, e.message);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e, stack) {
      debugPrint('[AuthService] Unexpected signIn error: $e\n$stack');
      _errorMessage = 'Authentication error: ${e.toString()}';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  /// Create new user account with email and password
  Future<bool> signUpWithEmailPassword(String email, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      debugPrint('[AuthService] Attempting createUser for: ${email.trim()}');
      final credential = await _auth.createUserWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
      _currentUser = credential.user;
      _isDemoMode = false;
      _isLoading = false;
      _errorMessage = null;
      debugPrint('[AuthService] User registered successfully: UID=${credential.user?.uid}');

      // PART 2: Optional Firestore user document creation
      // Note: Must not cause signup to fail if Firestore rules or network are offline
      if (credential.user != null) {
        try {
          await FirebaseFirestore.instance
              .collection('users')
              .doc(credential.user!.uid)
              .set({
            'uid': credential.user!.uid,
            'email': credential.user!.email,
            'createdAt': FieldValue.serverTimestamp(),
          }, SetOptions(merge: true));
          debugPrint('[AuthService] Firestore user profile stored: users/${credential.user!.uid}');
        } catch (firestoreError) {
          debugPrint('[AuthService] Firestore profile creation notice (Auth succeeded independently): $firestoreError');
        }
      }

      notifyListeners();
      return true;
    } on FirebaseAuthException catch (e) {
      debugPrint('[AuthService] FirebaseAuthException during signup: code=${e.code}, message=${e.message}');
      _errorMessage = _getReadableAuthError(e.code, e.message);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e, stack) {
      debugPrint('[AuthService] Unexpected signup error: $e\n$stack');
      _errorMessage = 'Registration error: ${e.toString()}';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  /// Demo Login option (useful for college project testing without live network)
  void loginAsDemoUser({String email = 'deaf.assist@tinyml-soundalert.org'}) {
    _isDemoMode = true;
    _demoEmail = email;
    _currentUser = null;
    _errorMessage = null;
    notifyListeners();
  }

  /// Sign out
  Future<void> signOut() async {
    try {
      await _auth.signOut();
    } catch (e) {
      debugPrint('[AuthService] SignOut note: $e');
    }
    _currentUser = null;
    _isDemoMode = false;
    _demoEmail = null;
    _errorMessage = null;
    notifyListeners();
  }

  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  String _getReadableAuthError(String code, String? defaultMsg) {
    switch (code) {
      case 'email-already-in-use':
        return 'This email is already in use by another account. Please log in or use a different email.';
      case 'invalid-email':
        return 'The email address is not properly formatted. Please enter a valid email.';
      case 'weak-password':
        return 'The password is too weak. Please use at least 6 characters.';
      case 'operation-not-allowed':
        return 'Email/Password sign-in is not enabled in Firebase Console. Please enable Email/Password provider.';
      case 'network-request-failed':
        return 'Network connection failed. Please check your internet connection and try again.';
      case 'too-many-requests':
        return 'Too many requests. Access is temporarily restricted to protect your account. Please try again later.';
      case 'user-disabled':
        return 'This account has been disabled. Please contact system support.';
      case 'user-not-found':
        return 'No registered account found with this email address.';
      case 'wrong-password':
      case 'invalid-credential':
        return 'Incorrect email or password. Please check your credentials.';
      case 'channel-error':
        return 'Please ensure both email and password fields are completed.';
      default:
        // Prevent displaying a bare generic "Error" message
        if (defaultMsg == null || defaultMsg.trim().isEmpty || defaultMsg.trim() == 'Error') {
          return 'Authentication failed ($code). Please check your connection and credentials.';
        }
        return defaultMsg;
    }
  }
}

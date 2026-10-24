import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import '../models/sound_detection.dart';

class FirestoreAlertService extends ChangeNotifier {
  final FirebaseFirestore? _firestoreOverride;
  FirebaseFirestore get _firestore => _firestoreOverride ?? FirebaseFirestore.instance;

  // In-memory fallback list to ensure zero app crashes if offline or demo mode
  final List<SoundDetection> _localCache = [];

  FirestoreAlertService({FirebaseFirestore? firestore}) : _firestoreOverride = firestore;

  FirebaseFirestore get firestore => _firestore;

  /// Firestore sub-collection reference: users/{userId}/detections
  CollectionReference<Map<String, dynamic>> _detectionsRef(String userId) {
    return _firestore.collection('users').doc(userId).collection('detections');
  }

  /// Store a valid sound detection event in Cloud Firestore
  Future<String?> saveDetection(SoundDetection detection, {required String userId}) async {
    // Keep local copy immediately
    _localCache.insert(0, detection);
    notifyListeners();

    try {
      final docRef = await _detectionsRef(userId).add(
        detection.toFirestoreMap(useServerTimestamp: true),
      );
      debugPrint('Detection saved to Firestore with ID: ${docRef.id}');
      return docRef.id;
    } catch (e) {
      debugPrint('Firestore save error (using local cache): $e');
      return null;
    }
  }

  /// Stream detection history from Firestore with optional sound type & date filtering
  Stream<List<SoundDetection>> getDetectionsStream(
    String userId, {
    String? soundFilter,
    DateTime? dateFilter,
  }) {
    try {
      Query<Map<String, dynamic>> query = _detectionsRef(userId)
          .orderBy('timestamp', descending: true)
          .limit(100);

      if (soundFilter != null && soundFilter.isNotEmpty && soundFilter != 'All') {
        query = query.where('sound', isEqualTo: soundFilter);
      }

      return query.snapshots().map((snapshot) {
        final list = snapshot.docs.map((doc) => SoundDetection.fromFirestore(doc)).toList();

        if (dateFilter != null) {
          return list.where((item) {
            return item.timestamp.year == dateFilter.year &&
                item.timestamp.month == dateFilter.month &&
                item.timestamp.day == dateFilter.day;
          }).toList();
        }

        return list;
      }).handleError((error) {
        debugPrint('Firestore stream error, falling back to cached events: $error');
        return _getLocalFiltered(soundFilter: soundFilter, dateFilter: dateFilter);
      });
    } catch (e) {
      debugPrint('Firestore stream init exception: $e');
      return Stream.value(_getLocalFiltered(soundFilter: soundFilter, dateFilter: dateFilter));
    }
  }

  List<SoundDetection> _getLocalFiltered({String? soundFilter, DateTime? dateFilter}) {
    return _localCache.where((item) {
      final matchesSound = soundFilter == null ||
          soundFilter.isEmpty ||
          soundFilter == 'All' ||
          item.sound.toLowerCase() == soundFilter.toLowerCase();
      final matchesDate = dateFilter == null ||
          (item.timestamp.year == dateFilter.year &&
              item.timestamp.month == dateFilter.month &&
              item.timestamp.day == dateFilter.day);
      return matchesSound && matchesDate;
    }).toList();
  }

  /// Count today's total alerts for the main dashboard
  Future<int> getTodayAlertCount(String userId) async {
    final now = DateTime.now();
    final startOfToday = DateTime(now.year, now.month, now.day);

    try {
      final snapshot = await _detectionsRef(userId)
          .where('timestamp', isGreaterThanOrEqualTo: Timestamp.fromDate(startOfToday))
          .get();

      return snapshot.docs.length;
    } catch (e) {
      debugPrint('Firestore count error, returning local cache count: $e');
      return _localCache.where((d) => d.timestamp.isAfter(startOfToday)).length;
    }
  }

  /// Clear detection history for the current user
  Future<bool> clearHistory(String userId) async {
    _localCache.clear();
    notifyListeners();

    try {
      final snapshot = await _detectionsRef(userId).limit(200).get();
      final batch = _firestore.batch();
      for (final doc in snapshot.docs) {
        batch.delete(doc.reference);
      }
      await batch.commit();
      return true;
    } catch (e) {
      debugPrint('Firestore clear history note: $e');
      return true; // Local was cleared
    }
  }
}

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../services/ble_service.dart';
import '../services/firebase_service.dart';
import '../models/sound_detection.dart';
import '../theme/app_theme.dart';
import '../widgets/connection_status.dart';
import '../widgets/sound_alert_card.dart';
import '../widgets/detection_history_card.dart';
import 'live_detection_screen.dart';
import 'history_screen.dart';
import 'settings_screen.dart';

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = const [
    DashboardScreen(),
    LiveDetectionScreen(),
    HistoryScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard_rounded),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.sensors_outlined),
            selectedIcon: Icon(Icons.sensors_rounded),
            label: 'Live BLE',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_rounded),
            selectedIcon: Icon(Icons.history_toggle_off_rounded),
            label: 'History',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings_rounded),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _todayCount = 0;

  @override
  void initState() {
    super.initState();
    _refreshTodayCount();
  }

  Future<void> _refreshTodayCount() async {
    final auth = Provider.of<AuthService>(context, listen: false);
    final firebase = Provider.of<FirestoreAlertService>(context, listen: false);
    final count = await firebase.getTodayAlertCount(auth.userId);
    if (mounted) {
      setState(() {
        _todayCount = count;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final bleService = context.watch<BleService>();
    final authService = context.watch<AuthService>();
    final firebaseService = context.watch<FirestoreAlertService>();

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF38BDF8).withValues(alpha: 0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.hearing_rounded, color: Color(0xFF38BDF8), size: 24),
            ),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: const [
                Text(
                  'Hearo',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 0.4,
                  ),
                ),
                Text(
                  'Every Sound Matters.',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF38BDF8),
                    letterSpacing: 0.2,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          // Connection Quick Indicator Pill
          Container(
            margin: const EdgeInsets.only(right: 12),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: bleService.isConnected
                  ? AppTheme.success.withValues(alpha: 0.15)
                  : AppTheme.critical.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: bleService.isConnected ? AppTheme.success : AppTheme.critical,
                width: 1.2,
              ),
            ),
            child: Row(
              children: [
                Icon(
                  bleService.isConnected ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
                  color: bleService.isConnected ? AppTheme.success : AppTheme.critical,
                  size: 16,
                ),
                const SizedBox(width: 6),
                Text(
                  bleService.isConnected ? 'CONNECTED' : 'DISCONNECTED',
                  style: TextStyle(
                    color: bleService.isConnected ? AppTheme.success : AppTheme.critical,
                    fontWeight: FontWeight.bold,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _refreshTodayCount,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.symmetric(vertical: 12.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Hardware & TinyML State Card — now shows ESP32 and Microphone status separately
              ConnectionStatusCard(
                bleStatus: bleService.status,
                tinyMlStatus: bleService.tinyMlStatus,
                microphoneStatus: bleService.microphoneStatus,
                deviceName: bleService.deviceName,
              ),
              const SizedBox(height: 12),

              // Large, High-Contrast Visual Alert Area (Designed for Deaf Users)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'REAL-TIME SOUND ALERT',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: AppTheme.textSecondary,
                        letterSpacing: 1.1,
                      ),
                    ),
                    const SizedBox(height: 8),
                    SoundAlertCard(
                      detection: bleService.currentDetection,
                      isConnected: bleService.isConnected,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Daily Metric Statistics Banner
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Row(
                  children: [
                    // Total Alerts Today
                    Expanded(
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: AppTheme.surface,
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: const Color(0xFF334155), width: 1.5),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Row(
                              children: [
                                Icon(Icons.notifications_active_rounded, color: Color(0xFF38BDF8), size: 20),
                                SizedBox(width: 6),
                                Text(
                                  'Alerts Today',
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: AppTheme.textSecondary,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              '$_todayCount',
                              style: const TextStyle(
                                fontSize: 28,
                                fontWeight: FontWeight.w900,
                                color: AppTheme.textPrimary,
                              ),
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              'Logged to Firestore',
                              style: TextStyle(
                                fontSize: 11,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),

                    // Model & Microphone Profile Card
                    Expanded(
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: AppTheme.surface,
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: const Color(0xFF334155), width: 1.5),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Row(
                              children: [
                                Icon(Icons.mic_external_on_rounded, color: AppTheme.warning, size: 20),
                                SizedBox(width: 6),
                                Text(
                                  'INMP441 MEMS',
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: AppTheme.textSecondary,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              // Only show Active if ESP32 explicitly sent MIC_ON
                              bleService.isMicrophoneOn ? 'Active' : 'Standby',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.w900,
                                color: bleService.isMicrophoneOn
                                    ? AppTheme.success
                                    : AppTheme.inactive,
                              ),
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              'I2S Audio Sampling',
                              style: TextStyle(
                                fontSize: 11,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Recent Detections Preview
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'RECENT DETECTIONS',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: AppTheme.textSecondary,
                        letterSpacing: 1.1,
                      ),
                    ),
                    TextButton(
                      onPressed: () {
                        // Switch to history tab if needed or open history
                        DefaultTabController.maybeOf(context);
                      },
                      child: const Text('View All'),
                    ),
                  ],
                ),
              ),

              // Stream top 3 recent detections from Firestore
              StreamBuilder<List<SoundDetection>>(
                stream: firebaseService.getDetectionsStream(authService.userId),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting && !snapshot.hasData) {
                    return const Center(
                      child: Padding(
                        padding: EdgeInsets.all(24.0),
                        child: CircularProgressIndicator(),
                      ),
                    );
                  }

                  final detections = snapshot.data ?? [];
                  if (detections.isEmpty) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
                      child: Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: AppTheme.surface,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Center(
                          child: Text(
                            'No Hearo alerts logged yet today.\nIncoming BLE detections will appear here automatically.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: AppTheme.textMuted, fontSize: 14),
                          ),
                        ),
                      ),
                    );
                  }

                  final recentList = detections.take(3).toList();
                  return Column(
                    children: recentList.map((detection) {
                      return DetectionHistoryCard(detection: detection);
                    }).toList(),
                  );
                },
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

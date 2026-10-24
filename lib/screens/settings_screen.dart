import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/auth_service.dart';
import '../services/ble_service.dart';
import '../services/firebase_service.dart';
import '../services/notification_service.dart';
import '../models/sound_category.dart';
import '../theme/app_theme.dart';
import 'login_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _vibrationEnabled = true;
  bool _visualFlashEnabled = true;
  double _minConfidenceThreshold = 75.0;
  bool _dangerousNotificationsEnabled = NotificationService().notificationsEnabled;
  bool? _notificationPermissionGranted;

  @override
  void initState() {
    super.initState();
    _checkNotificationPermission();
  }

  Future<void> _checkNotificationPermission() async {
    final granted = await NotificationService().isPermissionGranted();
    if (mounted) {
      setState(() {
        _notificationPermissionGranted = granted;
      });
    }
  }

  Future<void> _requestNotificationPermission() async {
    final granted = await NotificationService().requestPermissions();
    if (mounted) {
      setState(() {
        _notificationPermissionGranted = granted;
      });
      if (!granted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: AppTheme.surfaceElevated,
            content: const Text(
              'Notification permission was denied. Please enable it in system settings for urgent alerts.',
              style: TextStyle(color: Colors.white),
            ),
            action: SnackBarAction(
              label: 'SETTINGS',
              textColor: AppTheme.primary,
              onPressed: () => openAppSettings(),
            ),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthService>();
    final ble = context.watch<BleService>();
    final firebase = context.watch<FirestoreAlertService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings & Configuration'),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
        children: [
          // Section 1: Bluetooth ESP32 Connection
          _buildSectionHeader('HARDWARE CONNECTION (ESP32-WROOM-32)'),
          Card(
            margin: EdgeInsets.zero,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        ble.isConnected ? Icons.bluetooth_connected : Icons.bluetooth_searching,
                        color: ble.isConnected ? AppTheme.success : AppTheme.primary,
                        size: 28,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              ble.isConnected ? 'Connected to Device' : 'BLE Scanner',
                              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                            ),
                            Text(
                              ble.isConnected ? ble.deviceName : 'Scan for ESP32-WROOM-32 BLE transmitter',
                              style: const TextStyle(color: AppTheme.textSecondary, fontSize: 13),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  if (ble.statusMessage != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12.0),
                      child: Text(
                        ble.statusMessage!,
                        style: TextStyle(
                          color: ble.isConnected ? AppTheme.success : AppTheme.warning,
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),

                  // Connect / Disconnect / Scan Action Buttons
                  Row(
                    children: [
                      if (ble.isConnected)
                        Expanded(
                          child: ElevatedButton.icon(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.critical,
                              foregroundColor: Colors.white,
                            ),
                            onPressed: () async {
                              await ble.disconnect();
                            },
                            icon: const Icon(Icons.bluetooth_disabled_rounded),
                            label: const Text('Disconnect'),
                          ),
                        )
                      else ...[
                        Expanded(
                          child: ElevatedButton.icon(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.primary,
                              foregroundColor: Colors.black,
                            ),
                            onPressed: ble.isScanning ? null : () => ble.startScan(),
                            icon: ble.isScanning
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black),
                                  )
                                : const Icon(Icons.radar_rounded),
                            label: Text(ble.isScanning ? 'Scanning...' : 'Scan Nearby BLE'),
                          ),
                        ),
                        if (ble.isScanning) ...[
                          const SizedBox(width: 8),
                          IconButton(
                            icon: const Icon(Icons.stop_circle_outlined, color: AppTheme.warning),
                            onPressed: () => ble.stopScan(),
                          ),
                        ],
                      ],
                    ],
                  ),

                  // Scan Results List if scanning and not connected
                  if (!ble.isConnected && ble.scanResults.isNotEmpty) ...[
                    const Divider(color: Color(0xFF334155), height: 28),
                    const Text(
                      'Discovered BLE Devices:',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                    ),
                    const SizedBox(height: 8),
                    ...ble.scanResults.map((result) {
                      final name = result.device.platformName.isNotEmpty
                          ? result.device.platformName
                          : (result.advertisementData.advName.isNotEmpty
                              ? result.advertisementData.advName
                              : 'Hearo Device');
                      final isEsp = name.toLowerCase().contains('esp') ||
                          name.toLowerCase().contains('sound') ||
                          name.toLowerCase().contains('tinyml');

                      return ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(
                          Icons.bluetooth,
                          color: isEsp ? AppTheme.primary : AppTheme.textMuted,
                        ),
                        title: Text(
                          name,
                          style: TextStyle(
                            fontWeight: isEsp ? FontWeight.bold : FontWeight.normal,
                            color: isEsp ? AppTheme.primary : AppTheme.textPrimary,
                          ),
                        ),
                        subtitle: Text('RSSI: ${result.rssi} dBm • ID: ${result.device.remoteId}'),
                        trailing: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(90, 36),
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                          ),
                          onPressed: () => ble.connectToDevice(result.device),
                          child: const Text('Connect', style: TextStyle(fontSize: 13)),
                        ),
                      );
                    }),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Section 2: Supported Sound Categories Directory (20 TinyML Classes)
          _buildSectionHeader('SUPPORTED SOUND CATEGORIES (20 CLASSES)'),
          Card(
            margin: EdgeInsets.zero,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'The TinyML model running on the ESP32-WROOM-32 classifies acoustic signatures into 20 sound categories.',
                    style: TextStyle(fontSize: 13, color: AppTheme.textSecondary, height: 1.4),
                  ),
                  const SizedBox(height: 14),

                  // Risk Level Legend
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0F172A),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFF334155), width: 1),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'RISK LEVEL CLASSIFICATION LEGEND',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            color: AppTheme.textMuted,
                            letterSpacing: 0.8,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Wrap(
                          spacing: 8,
                          runSpacing: 6,
                          children: [
                            _buildLegendPill(
                              label: 'DANGEROUS',
                              color: AppTheme.critical,
                              description: 'Emergency Hazard',
                            ),
                            _buildLegendPill(
                              label: 'MEDIUM',
                              color: AppTheme.warning,
                              description: 'Important Sound',
                            ),
                            _buildLegendPill(
                              label: 'SAFE',
                              color: AppTheme.success,
                              description: 'Normal Ambient',
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // DANGEROUS Category Section
                  _buildRiskCategoryGroup(
                    title: 'DANGEROUS SOUNDS',
                    badgeText: 'DANGEROUS',
                    badgeColor: AppTheme.critical,
                    headerIcon: Icons.warning_rounded,
                    categories: SoundCategoryRegistry.dangerousCategories,
                  ),
                  const SizedBox(height: 14),

                  // MEDIUM Category Section
                  _buildRiskCategoryGroup(
                    title: 'MEDIUM SOUNDS',
                    badgeText: 'MEDIUM',
                    badgeColor: AppTheme.warning,
                    headerIcon: Icons.notifications_active_rounded,
                    categories: SoundCategoryRegistry.mediumCategories,
                  ),
                  const SizedBox(height: 14),

                  // SAFE Category Section
                  _buildRiskCategoryGroup(
                    title: 'SAFE SOUNDS',
                    badgeText: 'SAFE',
                    badgeColor: AppTheme.success,
                    headerIcon: Icons.check_circle_rounded,
                    categories: SoundCategoryRegistry.safeCategories,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Section 3: Accessibility & Sensory Alert Preferences
          _buildSectionHeader('ACCESSIBILITY & SAFETY NOTIFICATIONS'),
          Card(
            margin: EdgeInsets.zero,
            child: Column(
              children: [
                SwitchListTile(
                  secondary: const Icon(Icons.notification_important_rounded, color: AppTheme.critical),
                  title: const Text('Dangerous Sound Safety Alerts', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Instant high-priority notifications for sirens, alarms, glass breaking, car horns, screaming, etc.'),
                  value: _dangerousNotificationsEnabled,
                  onChanged: (val) {
                    setState(() {
                      _dangerousNotificationsEnabled = val;
                      NotificationService().notificationsEnabled = val;
                    });
                  },
                ),
                const Divider(color: Color(0xFF334155), height: 1),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
                  child: Row(
                    children: [
                      Icon(
                        _notificationPermissionGranted == true
                            ? Icons.verified_user_rounded
                            : Icons.warning_amber_rounded,
                        color: _notificationPermissionGranted == true
                            ? AppTheme.success
                            : AppTheme.warning,
                        size: 24,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _notificationPermissionGranted == true
                                  ? 'System Notification Permission: Active'
                                  : 'Notification Permission: Required',
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                fontSize: 13,
                                color: _notificationPermissionGranted == true
                                    ? AppTheme.textPrimary
                                    : AppTheme.warning,
                              ),
                            ),
                            Text(
                              _notificationPermissionGranted == true
                                  ? 'Channel "Hearo Safety Alerts" is ready for real-time background notifications.'
                                  : 'Android requires notification permission to alert you when Hearo is in background.',
                              style: const TextStyle(fontSize: 11, color: AppTheme.textMuted),
                            ),
                          ],
                        ),
                      ),
                      if (_notificationPermissionGranted != true) ...[
                        const SizedBox(width: 8),
                        TextButton(
                          onPressed: _requestNotificationPermission,
                          style: TextButton.styleFrom(
                            foregroundColor: AppTheme.primary,
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          ),
                          child: const Text('ENABLE', style: TextStyle(fontWeight: FontWeight.w900, fontSize: 12)),
                        ),
                      ],
                    ],
                  ),
                ),
                const Divider(color: Color(0xFF334155), height: 1),
                SwitchListTile(
                  secondary: const Icon(Icons.vibration_rounded, color: AppTheme.primary),
                  title: const Text('Haptic Vibration Alerts', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Strong tactile vibration pulses on incoming sound alerts'),
                  value: _vibrationEnabled,
                  onChanged: (val) {
                    setState(() {
                      _vibrationEnabled = val;
                    });
                  },
                ),
                const Divider(color: Color(0xFF334155), height: 1),
                SwitchListTile(
                  secondary: const Icon(Icons.flash_on_rounded, color: AppTheme.warning),
                  title: const Text('High-Contrast Strobe Glow', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Intense visual pulsing banner for critical alarms'),
                  value: _visualFlashEnabled,
                  onChanged: (val) {
                    setState(() {
                      _visualFlashEnabled = val;
                    });
                  },
                ),
                const Divider(color: Color(0xFF334155), height: 1),
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Confidence Threshold:',
                            style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
                          ),
                          Text(
                            '${_minConfidenceThreshold.round()}%',
                            style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.primary),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Slider(
                        value: _minConfidenceThreshold,
                        min: 50.0,
                        max: 95.0,
                        divisions: 9,
                        activeColor: AppTheme.primary,
                        onChanged: (val) {
                          setState(() {
                            _minConfidenceThreshold = val;
                          });
                        },
                      ),
                      const Text(
                        'Alerts below this confidence will be filtered out to prevent false alarms.',
                        style: TextStyle(fontSize: 12, color: AppTheme.textMuted),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Section 4: Firebase Account & Cloud Sync
          _buildSectionHeader('FIREBASE & ACCOUNT'),
          Card(
            margin: EdgeInsets.zero,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.cloud_done_rounded, color: AppTheme.success, size: 24),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              auth.userEmail,
                              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                            ),
                            Text(
                              'User ID: ${auth.userId.substring(0, auth.userId.length > 16 ? 16 : auth.userId.length)}...',
                              style: const TextStyle(fontSize: 12, color: AppTheme.textMuted),
                            ),
                          ],
                        ),
                      ),
                      if (auth.isDemoMode)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: AppTheme.warning.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: const Text('DEMO', style: TextStyle(color: AppTheme.warning, fontSize: 11, fontWeight: FontWeight.bold)),
                        ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.critical,
                      side: const BorderSide(color: AppTheme.critical),
                    ),
                    onPressed: () async {
                      final confirmed = await showDialog<bool>(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          backgroundColor: AppTheme.surface,
                          title: const Text('Clear History'),
                          content: const Text('Do you want to delete all alert history from Cloud Firestore?'),
                          actions: [
                            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                            ElevatedButton(
                              style: ElevatedButton.styleFrom(backgroundColor: AppTheme.critical),
                              onPressed: () => Navigator.pop(ctx, true),
                              child: const Text('Clear'),
                            ),
                          ],
                        ),
                      );
                      if (confirmed == true) {
                        await firebase.clearHistory(auth.userId);
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('History cleared successfully')),
                          );
                        }
                      }
                    },
                    icon: const Icon(Icons.delete_outline_rounded),
                    label: const Text('Clear Detection History'),
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF334155),
                      foregroundColor: Colors.white,
                    ),
                    onPressed: () async {
                      await ble.disconnect();
                      await auth.signOut();
                      if (context.mounted) {
                        Navigator.of(context).pushAndRemoveUntil(
                          MaterialPageRoute(builder: (_) => const LoginScreen()),
                          (route) => false,
                        );
                      }
                    },
                    icon: const Icon(Icons.logout_rounded),
                    label: const Text('Logout'),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Section 5: About Hearo
          _buildSectionHeader('ABOUT HEARO'),
          Card(
            margin: EdgeInsets.zero,
            child: Padding(
              padding: const EdgeInsets.all(18.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: AppTheme.primary.withValues(alpha: 0.15),
                          border: Border.all(color: AppTheme.primary, width: 1.5),
                        ),
                        child: const Icon(Icons.hearing_rounded, color: AppTheme.primary, size: 28),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Hearo',
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w900,
                                color: AppTheme.textPrimary,
                                letterSpacing: 0.5,
                              ),
                            ),
                            const Text(
                              'Every Sound Matters.',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: AppTheme.primary,
                              ),
                            ),
                            const SizedBox(height: 2),
                            const Text(
                              'Version 1.0.0 (Build 1)',
                              style: TextStyle(fontSize: 11, color: AppTheme.textMuted),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  const Text(
                    'Hearo is an assistive-technology platform designed for deaf and hard-of-hearing individuals. It connects over Bluetooth Low Energy to an ESP32-WROOM-32 smart sensor running an on-device Edge Impulse TinyML model that classifies 20 environmental sound categories and instantly triggers safety notifications for hazards.',
                    style: TextStyle(fontSize: 12.5, height: 1.5, color: AppTheme.textSecondary),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppTheme.background,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFF334155)),
                    ),
                    child: Column(
                      children: [
                        _buildAboutDetailRow(Icons.memory_rounded, 'Microcontroller', 'ESP32-WROOM-32'),
                        const Divider(color: Color(0xFF334155), height: 14),
                        _buildAboutDetailRow(Icons.mic_external_on_rounded, 'Acoustic Sensor', 'INMP441 MEMS (I2S)'),
                        const Divider(color: Color(0xFF334155), height: 14),
                        _buildAboutDetailRow(Icons.psychology_rounded, 'TinyML Model', '20-Class Acoustic Classifier'),
                        const Divider(color: Color(0xFF334155), height: 14),
                        _buildAboutDetailRow(Icons.bluetooth_rounded, 'BLE Interface', 'Custom UUID Stream'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0, left: 4.0),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w800,
          color: AppTheme.textSecondary,
          letterSpacing: 1.0,
        ),
      ),
    );
  }

  Widget _buildLegendPill({
    required String label,
    required Color color,
    required String description,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 1.2),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w900,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '• $description',
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 10,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskCategoryGroup({
    required String title,
    required String badgeText,
    required Color badgeColor,
    required IconData headerIcon,
    required List<SoundCategory> categories,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A).withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: badgeColor.withValues(alpha: 0.4), width: 1.2),
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(headerIcon, color: badgeColor, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '$title (${categories.length})',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w800,
                    color: badgeColor,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: badgeColor.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: badgeColor, width: 1),
                ),
                child: Text(
                  badgeText,
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w900,
                    color: badgeColor,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Divider(color: Color(0xFF1E293B), height: 1),
          const SizedBox(height: 6),
          ...categories.map((cat) => _buildCategoryRow(cat)),
        ],
      ),
    );
  }

  Widget _buildCategoryRow(SoundCategory cat) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5.0),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: cat.riskLevel.backgroundColor,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: cat.riskLevel.color.withValues(alpha: 0.4), width: 1),
            ),
            child: Icon(cat.icon, color: cat.riskLevel.color, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  cat.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: AppTheme.textPrimary,
                  ),
                ),
                Text(
                  cat.id,
                  style: const TextStyle(
                    fontSize: 11,
                    fontFamily: 'Courier',
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: cat.riskLevel.backgroundColor,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: cat.riskLevel.color, width: 1),
            ),
            child: Text(
              cat.riskLevel.label,
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w900,
                color: cat.riskLevel.color,
                letterSpacing: 0.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAboutDetailRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 16, color: AppTheme.primary),
        const SizedBox(width: 10),
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: AppTheme.textSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(
            fontSize: 12,
            color: AppTheme.textPrimary,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }
}

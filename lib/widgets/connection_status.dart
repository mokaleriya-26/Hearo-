import 'package:flutter/material.dart';
import '../services/ble_service.dart';
import '../theme/app_theme.dart';

/// Displays two independent device status indicators:
///   • ESP32 connection status (from actual BLE connection state)
///   • Microphone status (from ESP32-sent MIC_ON / MIC_OFF messages only)
///
/// Designed for deaf/hard-of-hearing users: every state uses text + icon + color.
class ConnectionStatusCard extends StatelessWidget {
  final BleStatus bleStatus;
  final TinyMlStatus tinyMlStatus;
  final MicrophoneStatus microphoneStatus;
  final String deviceName;
  final VoidCallback? onTapAction;

  const ConnectionStatusCard({
    super.key,
    required this.bleStatus,
    required this.tinyMlStatus,
    required this.microphoneStatus,
    this.deviceName = 'ESP32-WROOM-32',
    this.onTapAction,
  });

  @override
  Widget build(BuildContext context) {
    final bool isConnected = bleStatus == BleStatus.connected;
    final bool isConnecting = bleStatus == BleStatus.connecting;

    // ── ESP32 Connection Status ──────────────────────────────────────────────
    final Color esp32Color = isConnected
        ? AppTheme.success
        : (isConnecting ? AppTheme.warning : AppTheme.critical);

    final IconData esp32Icon = isConnected
        ? Icons.bluetooth_connected_rounded
        : (isConnecting
            ? Icons.bluetooth_searching_rounded
            : Icons.bluetooth_disabled_rounded);

    final String esp32Label = isConnected
        ? 'Connected'
        : (isConnecting ? 'Connecting...' : 'Disconnected');

    final String esp32SubLabel = isConnected
        ? deviceName
        : (isConnecting ? 'Establishing BLE link...' : 'ESP32 not detected');

    // ── Microphone Status ────────────────────────────────────────────────────
    final Color micColor;
    final IconData micIcon;
    final String micLabel;
    final String micSubLabel;

    switch (microphoneStatus) {
      case MicrophoneStatus.on:
        micColor = AppTheme.success;
        micIcon = Icons.mic_rounded;
        micLabel = 'Microphone ON';
        micSubLabel = 'Listening for sounds';
      case MicrophoneStatus.off:
        micColor = AppTheme.inactive;
        micIcon = Icons.mic_off_rounded;
        micLabel = 'Microphone OFF';
        micSubLabel = 'Sound detection stopped';
      case MicrophoneStatus.unavailable:
        micColor = AppTheme.inactive;
        micIcon = Icons.mic_off_rounded;
        micLabel = 'Microphone Unavailable';
        micSubLabel = 'Connect ESP32 first';
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Section header label
          const Text(
            'DEVICE STATUS',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              color: AppTheme.textSecondary,
              letterSpacing: 1.1,
            ),
          ),
          const SizedBox(height: 10),

          // Card containing both status rows
          Container(
            decoration: BoxDecoration(
              color: AppTheme.surface,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0xFF334155), width: 1.5),
            ),
            child: Column(
              children: [
                // ── ESP32 Status Row ─────────────────────────────────────────
                _StatusRow(
                  icon: esp32Icon,
                  statusColor: esp32Color,
                  deviceLabel: 'ESP32-WROOM-32',
                  statusLabel: esp32Label,
                  subLabel: esp32SubLabel,
                  isTopRow: true,
                ),

                const Divider(
                  color: Color(0xFF1E293B),
                  height: 1,
                  indent: 16,
                  endIndent: 16,
                ),

                // ── Microphone Status Row ────────────────────────────────────
                _StatusRow(
                  icon: micIcon,
                  statusColor: micColor,
                  deviceLabel: 'INMP441 Microphone',
                  statusLabel: micLabel,
                  subLabel: micSubLabel,
                  isTopRow: false,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// A single device status row with icon, label, status text, sub-label, and
/// a high-contrast indicator dot. Uses text + icon + color for accessibility.
class _StatusRow extends StatelessWidget {
  final IconData icon;
  final Color statusColor;
  final String deviceLabel;
  final String statusLabel;
  final String subLabel;
  final bool isTopRow;

  const _StatusRow({
    required this.icon,
    required this.statusColor,
    required this.deviceLabel,
    required this.statusLabel,
    required this.subLabel,
    required this.isTopRow,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        left: 16,
        right: 16,
        top: isTopRow ? 18 : 14,
        bottom: isTopRow ? 14 : 18,
      ),
      child: Row(
        children: [
          // Icon circle with colored background
          Container(
            padding: const EdgeInsets.all(11),
            decoration: BoxDecoration(
              color: statusColor.withValues(alpha: 0.15),
              shape: BoxShape.circle,
              border: Border.all(
                color: statusColor.withValues(alpha: 0.5),
                width: 1.5,
              ),
            ),
            child: Icon(icon, color: statusColor, size: 22),
          ),
          const SizedBox(width: 14),

          // Text block
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Device type name (small, secondary color)
                Text(
                  deviceLabel,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.textMuted,
                    letterSpacing: 0.3,
                  ),
                ),
                const SizedBox(height: 3),
                // Primary status label (large, colored, high-contrast)
                Text(
                  statusLabel,
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                  ),
                ),
                const SizedBox(height: 2),
                // Sub-label (descriptive text)
                Text(
                  subLabel,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppTheme.textSecondary,
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          // Glowing indicator dot (color-coded)
          Container(
            width: 14,
            height: 14,
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: statusColor.withValues(alpha: 0.5),
                  blurRadius: 6,
                  spreadRadius: 1,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

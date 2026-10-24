import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/ble_service.dart';
import '../theme/app_theme.dart';
import '../models/sound_category.dart';

class LiveDetectionScreen extends StatefulWidget {
  const LiveDetectionScreen({super.key});

  @override
  State<LiveDetectionScreen> createState() => _LiveDetectionScreenState();
}

class _LiveDetectionScreenState extends State<LiveDetectionScreen> {
  // Demo simulation options for presentation/debugging
  String _selectedSimulatedSound = 'Doorbell';
  double _simulatedConfidence = 94.5;

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleService>();
    final current = ble.currentDetection;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live BLE Stream'),
        actions: [
          IconButton(
            tooltip: 'Debug Test Packet',
            icon: const Icon(Icons.bug_report_outlined),
            onPressed: () => _showSimulateDialog(context, ble),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Live Status Header Banner
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: ble.isConnected
                    ? AppTheme.success.withValues(alpha: 0.12)
                    : AppTheme.critical.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: ble.isConnected ? AppTheme.success : AppTheme.critical,
                  width: 1.5,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    ble.isConnected ? Icons.bluetooth_connected_rounded : Icons.bluetooth_disabled_rounded,
                    color: ble.isConnected ? AppTheme.success : AppTheme.critical,
                    size: 32,
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          ble.isConnected ? 'CONNECTED TO ESP32' : 'ESP32 DISCONNECTED',
                          style: TextStyle(
                            color: ble.isConnected ? AppTheme.success : AppTheme.critical,
                            fontSize: 15,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 0.5,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Device: ${ble.deviceName}',
                          style: const TextStyle(
                            color: AppTheme.textPrimary,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Parsed Live Data Cards
            const Text(
              'PARSED BLE TELEMETRY',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w800,
                color: AppTheme.textSecondary,
                letterSpacing: 1.0,
              ),
            ),
            const SizedBox(height: 10),

            // Sound Card
            _buildTelemetryRow(
              icon: Icons.graphic_eq_rounded,
              title: 'Sound Detected',
              value: current?.sound ?? (ble.isConnected ? 'Waiting for sound...' : 'No data'),
              highlight: current != null,
              accentColor: current?.category.primaryColor ?? AppTheme.primary,
            ),
            const SizedBox(height: 10),

            // Confidence Card
            _buildTelemetryRow(
              icon: Icons.analytics_outlined,
              title: 'Confidence',
              value: current != null ? current.confidenceDisplay : (ble.isConnected ? '-- %' : 'N/A'),
              highlight: current != null,
              accentColor: const Color(0xFF10B981),
            ),
            const SizedBox(height: 10),

            // Time Card
            _buildTelemetryRow(
              icon: Icons.schedule_rounded,
              title: 'Detection Time',
              value: current != null ? current.formattedTime : (ble.isConnected ? '--:--' : 'N/A'),
              highlight: current != null,
              accentColor: const Color(0xFFF59E0B),
            ),
            const SizedBox(height: 24),

            // Last Received Raw BLE Message (JSON Console)
            const Text(
              'RAW BLE CHARACTERISTIC PAYLOAD',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w800,
                color: AppTheme.textSecondary,
                letterSpacing: 1.0,
              ),
            ),
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: const Color(0xFF020617), // Terminal / code dark background
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF334155), width: 1.5),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: const [
                          Icon(Icons.terminal_rounded, color: Color(0xFF38BDF8), size: 18),
                          SizedBox(width: 8),
                          Text(
                            'GATT Characteristic Notification',
                            style: TextStyle(
                              color: Color(0xFF94A3B8),
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1E293B),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text(
                          'UTF-8 JSON',
                          style: TextStyle(color: Color(0xFF38BDF8), fontSize: 11, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                  const Divider(color: Color(0xFF1E293B), height: 20),
                  SelectableText(
                    ble.lastReceivedRawMessage.isNotEmpty
                        ? ble.lastReceivedRawMessage
                        : (ble.isConnected
                            ? '{\n  "status": "awaiting_esp32_notification",\n  "service_uuid": "${BleService.serviceUuid}",\n  "char_uuid": "${BleService.characteristicUuid}"\n}'
                            : '{\n  "error": "ESP32 Not Connected",\n  "hint": "Please pair with ESP32-WROOM-32 in Settings"\n}'),
                    style: const TextStyle(
                      fontFamily: 'Courier',
                      fontSize: 14,
                      color: Color(0xFF38BDF8),
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // BLE Protocol Reference Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.surface,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF334155)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    'ESP32 TinyML JSON Protocol Reference:',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppTheme.textPrimary),
                  ),
                  SizedBox(height: 6),
                  Text(
                    'Format: {"sound": "Doorbell", "confidence": 94.5}\n'
                    'Characteristic: beb5483e-36e1-4688-b7f5-ea07361b26a8\n'
                    'Service: 4fafc201-1fb5-459e-8fcc-c5c9c331914b',
                    style: TextStyle(color: AppTheme.textSecondary, fontSize: 12, height: 1.4),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTelemetryRow({
    required IconData icon,
    required String title,
    required String value,
    required bool highlight,
    required Color accentColor,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: highlight ? accentColor.withValues(alpha: 0.6) : const Color(0xFF334155),
          width: 1.5,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: accentColor.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: accentColor, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppTheme.textMuted,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: highlight ? accentColor : AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showSimulateDialog(BuildContext context, BleService ble) {
    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            backgroundColor: AppTheme.surface,
            title: const Text('Simulate ESP32 BLE Packet'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Inject test JSON directly into the BLE parser to verify UI and sound alerts:',
                  style: TextStyle(fontSize: 13, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 16),
                const Text('Select Sound:', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                const SizedBox(height: 6),
                DropdownButtonFormField<String>(
                  initialValue: _selectedSimulatedSound,
                  dropdownColor: AppTheme.surface,
                  items: SoundCategoryRegistry.categories.map((c) {
                    return DropdownMenuItem(
                      value: c.name,
                      child: Text(c.displayName),
                    );
                  }).toList(),
                  onChanged: (val) {
                    if (val != null) {
                      setDialogState(() {
                        _selectedSimulatedSound = val;
                      });
                    }
                  },
                ),
                const SizedBox(height: 16),
                Text(
                  'Confidence: ${_simulatedConfidence.toStringAsFixed(1)}%',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                ),
                Slider(
                  value: _simulatedConfidence,
                  min: 50.0,
                  max: 100.0,
                  divisions: 50,
                  activeColor: AppTheme.primary,
                  onChanged: (val) {
                    setDialogState(() {
                      _simulatedConfidence = val;
                    });
                  },
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () {
                  ble.simulateBlePacketForTesting(
                    sound: _selectedSimulatedSound,
                    confidence: _simulatedConfidence,
                  );
                  Navigator.pop(ctx);
                },
                child: const Text('Inject Packet'),
              ),
            ],
          );
        },
      ),
    );
  }
}

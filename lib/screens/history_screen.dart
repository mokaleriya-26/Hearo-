import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../services/firebase_service.dart';
import '../models/sound_detection.dart';
import '../models/sound_category.dart';
import '../theme/app_theme.dart';
import '../widgets/detection_history_card.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  String _selectedSoundFilter = 'All';
  DateTime? _selectedDateFilter;

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedDateFilter ?? DateTime.now(),
      firstDate: DateTime(2023),
      lastDate: DateTime(2030),
      builder: (context, child) {
        return Theme(
          data: ThemeData.dark().copyWith(
            colorScheme: const ColorScheme.dark(
              primary: AppTheme.primary,
              surface: AppTheme.surface,
              onSurface: AppTheme.textPrimary,
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null) {
      setState(() {
        _selectedDateFilter = picked;
      });
    }
  }

  void _clearFilters() {
    setState(() {
      _selectedSoundFilter = 'All';
      _selectedDateFilter = null;
    });
  }

  Future<void> _confirmClearHistory(BuildContext context, FirestoreAlertService firebase, String userId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.surface,
        title: const Text('Clear All Detection History?'),
        content: const Text(
          'This will delete stored sound alert logs for this device from Cloud Firestore. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.critical),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Clear All'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await firebase.clearHistory(userId);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Detection history cleared')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthService>();
    final firebase = context.watch<FirestoreAlertService>();

    final List<String> soundOptions = [
      'All',
      ...SoundCategoryRegistry.categories.map((c) => c.name),
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection History'),
        actions: [
          IconButton(
            tooltip: 'Clear History',
            icon: const Icon(Icons.delete_sweep_rounded),
            onPressed: () => _confirmClearHistory(context, firebase, auth.userId),
          ),
        ],
      ),
      body: Column(
        children: [
          // Filter Chips Section (Sound Category)
          Container(
            padding: const EdgeInsets.only(top: 8, bottom: 4),
            child: SizedBox(
              height: 48,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 16),
                itemCount: soundOptions.length,
                separatorBuilder: (context, index) => const SizedBox(width: 8),
                itemBuilder: (context, index) {
                  final soundName = soundOptions[index];
                  final isSelected = _selectedSoundFilter == soundName;

                  return FilterChip(
                    label: Text(soundName),
                    selected: isSelected,
                    onSelected: (val) {
                      setState(() {
                        _selectedSoundFilter = val ? soundName : 'All';
                      });
                    },
                    selectedColor: AppTheme.primary.withValues(alpha: 0.25),
                    backgroundColor: AppTheme.surface,
                    checkmarkColor: AppTheme.primary,
                    labelStyle: TextStyle(
                      color: isSelected ? AppTheme.primary : AppTheme.textSecondary,
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                      fontSize: 13,
                    ),
                    side: BorderSide(
                      color: isSelected ? AppTheme.primary : const Color(0xFF334155),
                      width: 1.5,
                    ),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                  );
                },
              ),
            ),
          ),

          // Date Filter Toolbar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            child: Row(
              children: [
                // Date picker button
                OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    minimumSize: Size.zero,
                  ),
                  onPressed: _pickDate,
                  icon: const Icon(Icons.calendar_month_rounded, size: 18),
                  label: Text(
                    _selectedDateFilter == null
                        ? 'Filter by Date'
                        : DateFormat('dd MMM yyyy').format(_selectedDateFilter!),
                    style: const TextStyle(fontSize: 13),
                  ),
                ),
                if (_selectedDateFilter != null || _selectedSoundFilter != 'All') ...[
                  const SizedBox(width: 10),
                  TextButton.icon(
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 10),
                      minimumSize: Size.zero,
                    ),
                    onPressed: _clearFilters,
                    icon: const Icon(Icons.close_rounded, size: 16, color: AppTheme.warning),
                    label: const Text('Reset Filters', style: TextStyle(color: AppTheme.warning, fontSize: 13)),
                  ),
                ],
              ],
            ),
          ),
          const Divider(color: Color(0xFF334155), height: 1),

          // Real-time Firestore Stream List
          Expanded(
            child: StreamBuilder<List<SoundDetection>>(
              stream: firebase.getDetectionsStream(
                auth.userId,
                soundFilter: _selectedSoundFilter,
                dateFilter: _selectedDateFilter,
              ),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting && !snapshot.hasData) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (snapshot.hasError) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Text(
                        'Error loading history: ${snapshot.error}',
                        style: const TextStyle(color: AppTheme.critical),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  );
                }

                final detections = snapshot.data ?? [];

                if (detections.isEmpty) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(32.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.history_toggle_off_rounded,
                            size: 64,
                            color: AppTheme.inactive.withValues(alpha: 0.6),
                          ),
                          const SizedBox(height: 16),
                          const Text(
                            'No Detections Found',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: AppTheme.textPrimary,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _selectedSoundFilter != 'All' || _selectedDateFilter != null
                                ? 'No alerts match your active filters. Try resetting the filters.'
                                : 'When the ESP32-WROOM-32 detects sounds, records will be logged to Cloud Firestore and displayed here.',
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              fontSize: 14,
                              color: AppTheme.textSecondary,
                              height: 1.4,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }

                return ListView.builder(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  itemCount: detections.length,
                  itemBuilder: (context, index) {
                    final item = detections[index];
                    return DetectionHistoryCard(
                      detection: item,
                      onTap: () => _showDetailSheet(context, item),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  void _showDetailSheet(BuildContext context, SoundDetection item) {
    final category = item.category;

    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: category.primaryColor.withValues(alpha: 0.2),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(category.icon, color: category.primaryColor, size: 32),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          item.sound,
                          style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                        ),
                        Text(
                          category.displayName,
                          style: const TextStyle(color: AppTheme.textSecondary, fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const Divider(color: Color(0xFF334155), height: 28),
              _buildModalRow('Confidence:', item.confidenceDisplay),
              _buildModalRow('Date Logged:', item.formattedDate),
              _buildModalRow('Time Logged:', item.formattedTime),
              _buildModalRow('Source Device:', item.deviceId),
              _buildModalRow('Urgency Level:', category.isCritical ? 'CRITICAL ALERT' : 'Normal Notification'),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Close'),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildModalRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppTheme.textSecondary, fontSize: 15)),
          Text(value, style: const TextStyle(color: AppTheme.textPrimary, fontWeight: FontWeight.bold, fontSize: 15)),
        ],
      ),
    );
  }
}

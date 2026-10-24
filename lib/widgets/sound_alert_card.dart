import 'package:flutter/material.dart';
import '../models/sound_category.dart';
import '../models/sound_detection.dart';
import '../theme/app_theme.dart';

class SoundAlertCard extends StatefulWidget {
  final SoundDetection? detection;
  final bool isConnected;

  const SoundAlertCard({
    super.key,
    required this.detection,
    required this.isConnected,
  });

  @override
  State<SoundAlertCard> createState() => _SoundAlertCardState();
}

class _SoundAlertCardState extends State<SoundAlertCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.04).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isConnected) {
      return _buildDisconnectedState();
    }

    if (widget.detection == null) {
      return _buildIdleState();
    }

    final detection = widget.detection!;
    final category = detection.category;
    final isCritical = category.isCritical;
    final riskLevel = category.riskLevel;

    // Header banner icon and exact risk text
    final IconData badgeIcon;
    final String badgeLabel = riskLevel.label;
    switch (riskLevel) {
      case SoundRiskLevel.dangerous:
        badgeIcon = Icons.warning_amber_rounded;
      case SoundRiskLevel.medium:
        badgeIcon = Icons.notifications_active_rounded;
      case SoundRiskLevel.safe:
      case SoundRiskLevel.unknown:
        badgeIcon = Icons.check_circle_outline_rounded;
    }

    Widget cardContent = Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: category.primaryColor.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: category.primaryColor,
          width: isCritical ? 3.0 : 2.0,
        ),
        boxShadow: [
          BoxShadow(
            color: category.primaryColor.withValues(alpha: isCritical ? 0.35 : 0.15),
            blurRadius: isCritical ? 24 : 14,
            spreadRadius: isCritical ? 4 : 1,
          ),
        ],
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // Visual Urgency & Exact Risk Badge for Deaf Users
          Container(
            margin: const EdgeInsets.only(bottom: 14),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: category.primaryColor,
              borderRadius: BorderRadius.circular(30),
              boxShadow: [
                BoxShadow(
                  color: category.primaryColor.withValues(alpha: 0.4),
                  blurRadius: 8,
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(badgeIcon, color: category.onColor, size: 16),
                const SizedBox(width: 6),
                Text(
                  badgeLabel,
                  style: TextStyle(
                    color: category.onColor,
                    fontSize: 12,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 1.0,
                  ),
                ),
              ],
            ),
          ),

          // Enormous Accessible Icon
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: category.primaryColor.withValues(alpha: 0.25),
              shape: BoxShape.circle,
              border: Border.all(color: category.primaryColor, width: 2.5),
            ),
            child: Icon(
              category.icon,
              size: 56,
              color: category.primaryColor,
            ),
          ),
          const SizedBox(height: 14),

          // Sound Detected Title (Wraps nicely for long names without overflow)
          Text(
            '${detection.displayTitle.toUpperCase()} DETECTED',
            textAlign: TextAlign.center,
            softWrap: true,
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w900,
              color: category.primaryColor,
              letterSpacing: 0.8,
              height: 1.2,
            ),
          ),
          const SizedBox(height: 14),

          // Confidence Bar & Metric (Responsive and flexible)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: AppTheme.surfaceElevated.withValues(alpha: 0.7),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF475569)),
            ),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Flexible(
                      child: Text(
                        'AI Confidence',
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 15,
                          color: AppTheme.textSecondary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      detection.confidenceDisplay,
                      style: const TextStyle(
                        fontSize: 20,
                        color: AppTheme.textPrimary,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: (detection.confidence / 100).clamp(0.0, 1.0),
                    minHeight: 10,
                    backgroundColor: const Color(0xFF1E293B),
                    valueColor: AlwaysStoppedAnimation<Color>(category.primaryColor),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Detected Time & Date Pill (Responsive Wrap that fits any mobile width without overflow)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: AppTheme.surfaceElevated.withValues(alpha: 0.5),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFF334155), width: 1),
            ),
            child: Wrap(
              alignment: WrapAlignment.center,
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 14,
              runSpacing: 4,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.access_time_filled_rounded,
                      size: 15,
                      color: AppTheme.textSecondary,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      detection.formattedTime,
                      style: const TextStyle(
                        fontSize: 13,
                        color: AppTheme.textPrimary,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.calendar_today_rounded,
                      size: 14,
                      color: AppTheme.textMuted,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      detection.formattedDate,
                      style: const TextStyle(
                        fontSize: 13,
                        color: AppTheme.textSecondary,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );

    if (isCritical) {
      return ScaleTransition(
        scale: _pulseAnimation,
        child: cardContent,
      );
    }

    return cardContent;
  }

  Widget _buildDisconnectedState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 28),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF475569), width: 1.5),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: AppTheme.inactive.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.bluetooth_disabled_rounded,
              size: 52,
              color: AppTheme.inactive,
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'ESP32 Not Connected',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Connect your ESP32-WROOM-32 sound sensor in Settings to start receiving live sound alerts.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: AppTheme.textSecondary,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIdleState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 28),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF38BDF8).withValues(alpha: 0.4), width: 1.5),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF38BDF8).withValues(alpha: 0.12),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.hearing_rounded,
              size: 52,
              color: Color(0xFF38BDF8),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Sensor Active & Listening',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'The TinyML model is monitoring audio from the INMP441 MEMS microphone. Detected alerts will appear here instantly.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: AppTheme.textSecondary,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}

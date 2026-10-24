import 'package:flutter/material.dart';
import '../models/sound_detection.dart';
import '../theme/app_theme.dart';

class DetectionHistoryCard extends StatelessWidget {
  final SoundDetection detection;
  final VoidCallback? onTap;

  const DetectionHistoryCard({
    super.key,
    required this.detection,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final category = detection.category;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              // High Contrast Category Icon
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: category.primaryColor.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: category.primaryColor, width: 1.5),
                ),
                child: Icon(
                  category.icon,
                  color: category.primaryColor,
                  size: 28,
                ),
              ),
              const SizedBox(width: 16),

              // Title and Summary line
              // Example: Doorbell | 94% | 07 Oct 2026 | 02:35 PM
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Flexible(
                          child: Text(
                            detection.displayTitle,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 17,
                              fontWeight: FontWeight.bold,
                              color: AppTheme.textPrimary,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: category.riskLevel.backgroundColor,
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: category.riskLevel.color, width: 1),
                          ),
                          child: Text(
                            category.riskLevel.label,
                            style: TextStyle(
                              color: category.riskLevel.color,
                              fontSize: 10,
                              fontWeight: FontWeight.w900,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    // High-contrast clean string format
                    Text(
                      '${detection.confidence.round()}% conf  •  ${detection.formattedDate}  •  ${detection.formattedTime}',
                      style: const TextStyle(
                        fontSize: 13,
                        color: AppTheme.textSecondary,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),

              // Confidence percentage pill badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: category.primaryColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: category.primaryColor, width: 1),
                ),
                child: Text(
                  detection.confidenceDisplay,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: category.primaryColor,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

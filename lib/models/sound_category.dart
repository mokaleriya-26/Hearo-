import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Three distinct risk classifications for sound categories:
/// - DANGEROUS: Red badge and red accent (Emergency/immediate hazard)
/// - MEDIUM: Amber/orange badge and accent (Attention/important sound)
/// - SAFE: Green badge and accent (Normal/ambient sound)
enum SoundRiskLevel {
  dangerous,
  medium,
  safe,
  unknown;

  String get label {
    switch (this) {
      case SoundRiskLevel.dangerous:
        return 'DANGEROUS';
      case SoundRiskLevel.medium:
        return 'MEDIUM';
      case SoundRiskLevel.safe:
        return 'SAFE';
      case SoundRiskLevel.unknown:
        return 'SAFE';
    }
  }

  Color get color {
    switch (this) {
      case SoundRiskLevel.dangerous:
        return AppTheme.critical; // Red
      case SoundRiskLevel.medium:
        return AppTheme.warning;  // Amber/Orange
      case SoundRiskLevel.safe:
        return AppTheme.success;  // Green
      case SoundRiskLevel.unknown:
        return AppTheme.inactive;
    }
  }

  Color get backgroundColor {
    switch (this) {
      case SoundRiskLevel.dangerous:
        return const Color(0x33EF4444);
      case SoundRiskLevel.medium:
        return const Color(0x33F59E0B);
      case SoundRiskLevel.safe:
        return const Color(0x3310B981);
      case SoundRiskLevel.unknown:
        return const Color(0x2264748B);
    }
  }

  Color get onColor => this == SoundRiskLevel.medium ? Colors.black : Colors.white;
}

class SoundCategory {
  final String id;              // Exact firmware class identifier (e.g. 'bell_sounds')
  final String name;            // Human-readable name (e.g. 'Bell Sounds')
  final String displayName;     // Human-readable display title
  final SoundRiskLevel riskLevel;
  final IconData icon;
  final Color primaryColor;
  final Color onColor;
  final Color backgroundColor;
  final bool triggersVibration;
  final bool triggersFlash;

  const SoundCategory({
    required this.id,
    required this.name,
    String? displayName,
    required this.riskLevel,
    required this.icon,
    Color? primaryColor,
    Color? onColor,
    Color? backgroundColor,
    bool? triggersVibration,
    bool? triggersFlash,
  })  : displayName = displayName ?? name,
        primaryColor = primaryColor ?? (riskLevel == SoundRiskLevel.dangerous
            ? AppTheme.critical
            : (riskLevel == SoundRiskLevel.medium ? AppTheme.warning : AppTheme.success)),
        onColor = onColor ?? (riskLevel == SoundRiskLevel.medium ? Colors.black : Colors.white),
        backgroundColor = backgroundColor ??
            (riskLevel == SoundRiskLevel.dangerous
                ? const Color(0x33EF4444)
                : (riskLevel == SoundRiskLevel.medium ? const Color(0x33F59E0B) : const Color(0x3310B981))),
        triggersVibration = triggersVibration ?? (riskLevel != SoundRiskLevel.safe),
        triggersFlash = triggersFlash ?? (riskLevel == SoundRiskLevel.dangerous);

  bool get isCritical => riskLevel == SoundRiskLevel.dangerous;
}

/// Centralized configuration for all 20 TinyML acoustic classes supported by the firmware.
/// Maps exact model class names to human-readable labels, risk levels, and icons.
class SoundCategoryRegistry {
  static const List<SoundCategory> categories = [
    // 1. Bell Sounds (bell_sounds) -> MEDIUM
    SoundCategory(
      id: 'bell_sounds',
      name: 'Bell Sounds',
      displayName: 'Bell Sounds',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.notifications_active_rounded,
    ),
    // 2. Car Horn (car_horn) -> DANGEROUS
    SoundCategory(
      id: 'car_horn',
      name: 'Car Horn',
      displayName: 'Car Horn',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.directions_car_rounded,
    ),
    // 3. Cat (cat) -> SAFE
    SoundCategory(
      id: 'cat',
      name: 'Cat',
      displayName: 'Cat',
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.pets_rounded,
    ),
    // 4. Cheering (cheering) -> SAFE
    SoundCategory(
      id: 'cheering',
      name: 'Cheering',
      displayName: 'Cheering',
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.celebration_rounded,
    ),
    // 5. Clapping (clapping) -> SAFE
    SoundCategory(
      id: 'clapping',
      name: 'Clapping',
      displayName: 'Clapping',
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.thumb_up_rounded,
    ),
    // 6. Clock (clock) -> SAFE
    SoundCategory(
      id: 'clock',
      name: 'Clock',
      displayName: 'Clock',
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.access_time_rounded,
    ),
    // 7. Coughing (coughing) -> MEDIUM
    SoundCategory(
      id: 'coughing',
      name: 'Coughing',
      displayName: 'Coughing',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.sick_rounded,
    ),
    // 8. Crying/Sobbing (crying_sobbing) -> MEDIUM
    SoundCategory(
      id: 'crying_sobbing',
      name: 'Crying/Sobbing',
      displayName: 'Crying/Sobbing',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.sentiment_very_dissatisfied_rounded,
    ),
    // 9. Dog (dog) -> MEDIUM
    SoundCategory(
      id: 'dog',
      name: 'Dog',
      displayName: 'Dog',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.cruelty_free_rounded,
    ),
    // 10. Door Knock (door_knock) -> MEDIUM
    SoundCategory(
      id: 'door_knock',
      name: 'Door Knock',
      displayName: 'Door Knock',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.meeting_room_rounded,
    ),
    // 11. Doorbell (doorbell) -> MEDIUM
    SoundCategory(
      id: 'doorbell',
      name: 'Doorbell',
      displayName: 'Doorbell',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.doorbell_rounded,
    ),
    // 12. Explosive Sounds (explosive_sounds) -> DANGEROUS
    SoundCategory(
      id: 'explosive_sounds',
      name: 'Explosive Sounds',
      displayName: 'Explosive Sounds',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.local_fire_department_rounded,
    ),
    // 13. Footsteps (footsteps) -> MEDIUM
    SoundCategory(
      id: 'footsteps',
      name: 'Footsteps',
      displayName: 'Footsteps',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.directions_walk_rounded,
    ),
    // 14. General Alarm (general_alarm) -> DANGEROUS
    SoundCategory(
      id: 'general_alarm',
      name: 'General Alarm',
      displayName: 'General Alarm',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.alarm_on_rounded,
    ),
    // 15. Glass Breaking (glass_breaking) -> DANGEROUS
    SoundCategory(
      id: 'glass_breaking',
      name: 'Glass Breaking',
      displayName: 'Glass Breaking',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.broken_image_rounded,
    ),
    // 16. Laughter (laughter) -> SAFE
    SoundCategory(
      id: 'laughter',
      name: 'Laughter',
      displayName: 'Laughter',
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.sentiment_very_satisfied_rounded,
    ),
    // 17. Phone Ringtone (phone_ringtone) -> MEDIUM
    SoundCategory(
      id: 'phone_ringtone',
      name: 'Phone Ringtone',
      displayName: 'Phone Ringtone',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.phone_in_talk_rounded,
    ),
    // 18. Screaming (screaming) -> DANGEROUS
    SoundCategory(
      id: 'screaming',
      name: 'Screaming',
      displayName: 'Screaming',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.record_voice_over_rounded,
    ),
    // 19. Siren (siren) -> DANGEROUS
    SoundCategory(
      id: 'siren',
      name: 'Siren',
      displayName: 'Siren',
      riskLevel: SoundRiskLevel.dangerous,
      icon: Icons.emergency_rounded,
    ),
    // 20. Transport Sounds (transport_sounds) -> MEDIUM
    SoundCategory(
      id: 'transport_sounds',
      name: 'Transport Sounds',
      displayName: 'Transport Sounds',
      riskLevel: SoundRiskLevel.medium,
      icon: Icons.commute_rounded,
    ),
  ];

  /// Fallback representation for unrecognized acoustic classes.
  /// Never defaults to dangerous to prevent false emergency panics.
  static const SoundCategory defaultCategory = SoundCategory(
    id: 'unknown_sound',
    name: 'Unknown Sound',
    displayName: 'Unknown Sound',
    riskLevel: SoundRiskLevel.safe,
    icon: Icons.graphic_eq_rounded,
    primaryColor: AppTheme.inactive,
    onColor: Colors.white,
    backgroundColor: Color(0x2264748B),
    triggersVibration: false,
    triggersFlash: false,
  );

  /// Grouped categories by risk level
  static List<SoundCategory> get dangerousCategories =>
      categories.where((c) => c.riskLevel == SoundRiskLevel.dangerous).toList();

  static List<SoundCategory> get mediumCategories =>
      categories.where((c) => c.riskLevel == SoundRiskLevel.medium).toList();

  static List<SoundCategory> get safeCategories =>
      categories.where((c) => c.riskLevel == SoundRiskLevel.safe).toList();

  /// Resolve matching category by firmware class identifier or display name (case-insensitive)
  static SoundCategory getCategoryByName(String? name) {
    if (name == null || name.trim().isEmpty) {
      return defaultCategory;
    }
    final raw = name.trim();
    final lower = raw.toLowerCase();
    final normalized = lower
        .replaceAll(' ', '_')
        .replaceAll('/', '_')
        .replaceAll('-', '_');

    // 1. Direct match with exact firmware ID
    for (final cat in categories) {
      if (cat.id == normalized || cat.id == lower) {
        return cat;
      }
    }

    // 2. Direct match with display name
    for (final cat in categories) {
      if (cat.name.toLowerCase() == lower || cat.displayName.toLowerCase() == lower) {
        return cat;
      }
    }

    // 3. Normalized heuristic matching for backward compatibility
    for (final cat in categories) {
      if (normalized.contains(cat.id) || cat.id.contains(normalized)) {
        return cat;
      }
    }

    // Heuristics for legacy labels if passed from old test/database data
    if (normalized.contains('fire')) {
      return getCategoryByName('general_alarm');
    }
    if (normalized.contains('horn') || normalized.contains('vehicle')) {
      return getCategoryByName('car_horn');
    }
    if (normalized.contains('knock')) {
      return getCategoryByName('door_knock');
    }

    // Graceful fallback: return a safe unknown category without crashing
    return SoundCategory(
      id: normalized,
      name: raw,
      displayName: raw,
      riskLevel: SoundRiskLevel.safe,
      icon: Icons.graphic_eq_rounded,
      primaryColor: AppTheme.inactive,
      onColor: Colors.white,
      backgroundColor: const Color(0x2264748B),
      triggersVibration: false,
      triggersFlash: false,
    );
  }
}

# 🦻 TinyML Smart Sound Alert System for Deaf People

A real-time, accessibility-focused Flutter mobile application communicating via **Bluetooth Low Energy (BLE)** with an **ESP32-WROOM-32** microcontroller and **INMP441 MEMS microphone** running an on-device **TinyML acoustic classification model**. Detected environmental sounds are visually presented to deaf and hard-of-hearing users with high-contrast alerts and persisted to **Firebase Cloud Firestore**.

---

## 📱 Features & Highlights

- **Deaf-First Accessible UI**: High-contrast dark theme, ultra-large typography, clear category icons, pulsing urgency animations for critical alarms, and haptic vibration feedback.
- **Real-Time Bluetooth Low Energy (BLE)**: Scans, connects, and subscribes to notifications from the ESP32-WROOM-32 GATT service using `flutter_blue_plus`.
- **Standardized JSON Telemetry**: Clean UTF-8 JSON stream parsed dynamically on arrival.
- **Firebase Authentication**: Email and password authentication with instant demo mode for college presentations.
- **Cloud Firestore History**: Automatic logging of detections (`users/{userId}/detections/{detectionId}`) with sound type, AI confidence, date, time, and device ID.
- **History Filtering**: Real-time filtering by sound category and date with one-tap history clearing.
- **Clean Architecture**: Decoupled models, services, UI widgets, and theme with reactive Provider state management.

---

## 📂 Project Architecture

```
IOT Project/
├── lib/
│   ├── main.dart                      # App entry point, MultiProvider & BLE-to-Firestore pipeline
│   ├── firebase_options.dart          # Platform Firebase options template
│   ├── models/
│   │   ├── sound_detection.dart       # Detection model, JSON & Firestore serializers
│   │   └── sound_category.dart        # Configurable sound categories registry & styling (20 TinyML classes)
│   ├── services/
│   │   ├── auth_service.dart          # Firebase Authentication & demo fallback
│   │   ├── ble_service.dart           # FlutterBluePlus scanning, GATT notifications & parsing
│   │   └── firebase_service.dart      # Cloud Firestore CRUD & reactive streams (FirestoreAlertService)
│   ├── screens/
│   │   ├── splash_screen.dart         # Accessible branding splash screen
│   │   ├── login_screen.dart          # Sign in & demo presentation access
│   │   ├── signup_screen.dart         # New account registration
│   │   ├── dashboard_screen.dart      # Main dashboard with responsive alert card & today's count
│   │   ├── live_detection_screen.dart # Real-time telemetry, raw JSON console & packet tester
│   │   ├── history_screen.dart        # Firestore detection log with sound/date filters
│   │   └── settings_screen.dart       # BLE scanner, 20-class sound catalog & accessibility settings
│   ├── widgets/
│   │   ├── connection_status.dart     # Hardware & TinyML status badge card
│   │   ├── sound_alert_card.dart      # High-contrast visual sound alert card
│   │   └── detection_history_card.dart# History list item card
│   └── theme/
│       └── app_theme.dart             # High-contrast accessibility dark design tokens
├── firmware/
│   └── Soundsense_ESP32/              # ESP32-WROOM-32 Arduino C++ BLE firmware & I2S microphone driver
├── android/app/src/main/AndroidManifest.xml # Android BLE & Network permissions
├── ios/Runner/Info.plist              # iOS Bluetooth permission strings
└── pubspec.yaml                       # Flutter dependencies
```

---

## 🔊 Supported Sound Categories (20 TinyML Classes)

The TinyML model classifies acoustic signatures into 20 sound classes categorized into 3 risk levels:

| Sound Class | Firmware ID | Risk Level | Alert Color |
| :--- | :--- | :--- | :--- |
| **Bell Sounds** | `bell_sounds` | **MEDIUM** | Amber/Orange |
| **Car Horn** | `car_horn` | **DANGEROUS** | Red |
| **Cat** | `cat` | **SAFE** | Green |
| **Cheering** | `cheering` | **SAFE** | Green |
| **Clapping** | `clapping` | **SAFE** | Green |
| **Clock** | `clock` | **SAFE** | Green |
| **Coughing** | `coughing` | **MEDIUM** | Amber/Orange |
| **Crying/Sobbing** | `crying_sobbing` | **MEDIUM** | Amber/Orange |
| **Dog** | `dog` | **MEDIUM** | Amber/Orange |
| **Door Knock** | `door_knock` | **MEDIUM** | Amber/Orange |
| **Doorbell** | `doorbell` | **MEDIUM** | Amber/Orange |
| **Explosive Sounds** | `explosive_sounds` | **DANGEROUS** | Red |
| **Footsteps** | `footsteps` | **MEDIUM** | Amber/Orange |
| **General Alarm** | `general_alarm` | **DANGEROUS** | Red |
| **Glass Breaking** | `glass_breaking` | **DANGEROUS** | Red |
| **Laughter** | `laughter` | **SAFE** | Green |
| **Phone Ringtone** | `phone_ringtone` | **MEDIUM** | Amber/Orange |
| **Screaming** | `screaming` | **DANGEROUS** | Red |
| **Siren** | `siren` | **DANGEROUS** | Red |
| **Transport Sounds** | `transport_sounds` | **MEDIUM** | Amber/Orange |

---

## 📡 BLE GATT Configuration & Data Protocol

### 1. BLE UUIDs
- **Device Name:** `SoundSense-ESP32`
- **GATT Service UUID:** `4fafc201-1fb5-459e-8fcc-c5c9c331914b`
- **GATT Characteristic UUID:** `beb5483e-36e1-4688-b7f5-ea07361b26a8` (Properties: `READ`, `NOTIFY`)

### 2. JSON Notification Payload
Transmitted as a UTF-8 encoded string from the ESP32-WROOM-32:
```json
{
  "sound": "doorbell",
  "confidence": 94.5,
  "timestamp": 128450
}
```

---

## ☁️ Firebase Cloud Firestore Structure

### Collection Path:
```
users/{userId}/detections/{detectionId}
```

### Document Schema:
```json
{
  "sound": "doorbell",
  "confidence": 94.5,
  "timestamp": "FieldValue.serverTimestamp()",
  "deviceId": "ESP32-WROOM-32",
  "userId": "wXyZ1234abcd5678"
}
```

---

## ⚙️ Hardware Wiring (ESP32-WROOM-32 & INMP441)

| Hardware Component | ESP32-WROOM-32 Pin | Description |
| :--- | :--- | :--- |
| **INMP441 VDD** | **3V3** | 3.3V Power supply |
| **INMP441 GND** | **GND** | Ground |
| **INMP441 SD (DIN)** | **GPIO 32** | I2S Serial Data |
| **INMP441 WS (LRCK)** | **GPIO 15** | I2S Word Select (Clock) |
| **INMP441 SCK (BCLK)** | **GPIO 14** | I2S Continuous Clock |
| **INMP441 L/R** | **GND** | Left channel select |
| **RGB LED (Red)** | **GPIO 23** (via 220Ω) | Critical Alerts (Fire Alarm, Siren) |
| **RGB LED (Green)** | **GPIO 19** (via 220Ω) | BLE Connected Status / Safe |
| **RGB LED (Blue)** | **GPIO 18** (via 220Ω) | Doorbell / Knocking / Info |
| **Push Button** | **GPIO 4** (to GND) | Hardware alert test trigger |

*(Note: Buzzer, BC547 transistor, and 1k resistor are completely removed. The system is 100% visual and smartphone-based for deaf users).*

---

## 🧠 Replacing Test BLE Data with Real TinyML Inference

In [firmware/esp32_tinyml_ble_firmware.ino](file:///Users/riyamokale/Documents/Riya/IOT%20Project/firmware/esp32_tinyml_ble_firmware.ino):

1. **Train Model on Edge Impulse:**
   - Record sound clips for Doorbell, Fire Alarm, Horn, Knocking, Background noise.
   - Train an MFE/MFCC + 1D/2D CNN audio classification model.
   - Export as **Arduino Library** (`Your_TinyML_inferencing.h`).
2. **Include in Firmware:**
   ```cpp
   #include <Your_TinyML_inferencing.h>
   ```
3. **Capture I2S Audio Buffers into Signal:**
   ```cpp
   static int16_t sample_buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
   size_t bytes_read = 0;
   i2s_read(I2S_PORT, (char*)sample_buffer, sizeof(sample_buffer), &bytes_read, portMAX_DELAY);
   signal_t signal;
   signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
   signal.get_data = &raw_feature_get_data;
   ```
4. **Execute Classifier:**
   ```cpp
   ei_impulse_result_t result = { 0 };
   EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
   if (r == EI_IMPULSE_OK) {
       for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
           if (result.classification[ix].value > 0.75f) { // 75% threshold
               sendSoundDetectionAlert(
                   result.classification[ix].label,
                   result.classification[ix].value * 100.0f
               );
               break;
           }
       }
   }
   ```

---

## 🚀 Step-by-Step Instructions to Run the Application

### 1. Prerequisites
- Flutter SDK (≥ 3.20.0)
- Android Studio / Xcode / VS Code
- Arduino IDE with ESP32 board support (version ≥ 2.0.11 or 3.x)

### 2. Install Flutter Packages
```bash
flutter pub get
```

### 3. Verify Code Integrity
```bash
flutter analyze
```

### 4. Connect Firebase (Optional for Cloud Sync)
1. Go to [Firebase Console](https://console.firebase.google.com/) and create a project.
2. Enable **Authentication** (Email/Password).
3. Enable **Cloud Firestore** in test or production mode.
4. Run `flutterfire configure` or place `google-services.json` in `android/app/` and `GoogleService-Info.plist` in `ios/Runner/`.
> *Note: The app includes automatic fallback/demo login so you can run and present the application even without active Firebase credentials!*

### 5. Flash ESP32-WROOM-32 Firmware
1. Open [firmware/Soundsense_ESP32/Soundsense_ESP32.ino](file:///Users/riyamokale/Documents/Riya/IOT%20Project/firmware/Soundsense_ESP32/Soundsense_ESP32.ino) in Arduino IDE.
2. Select Board: **ESP32 Dev Module** (ESP32-WROOM-32).
3. Upload code to your ESP32-WROOM-32.

### 6. Run the Flutter App
```bash
flutter run
```

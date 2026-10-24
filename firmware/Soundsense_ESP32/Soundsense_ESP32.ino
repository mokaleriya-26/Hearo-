#include <Arduino.h>
#include <ESP_I2S.h>
#include <math.h>
#include <stdint.h>
#include <cstring>
#include <esp_heap_caps.h>
#include <TensorFlowLite_ESP32.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "soundsense_v9_lite_shifted_bn_model.h"

// =====================================================
// 1. HARDWARE PINS
// =====================================================

#define I2S_BCLK_PIN 26
#define I2S_WS_PIN   25
#define I2S_SD_PIN   33

#define LED_RED_PIN   18
#define LED_GREEN_PIN 19
#define LED_BLUE_PIN  21

#define BUTTON_PIN    27

// =====================================================
// BLE CONFIGURATION
// =====================================================

static const char* BLE_DEVICE_NAME = "SoundSense-ESP32";

static const char* BLE_SERVICE_UUID =
    "4fafc201-1fb5-459e-8fcc-c5c9c331914b";

static const char* BLE_CHARACTERISTIC_UUID =
    "beb5483e-36e1-4688-b7f5-ea07361b26a8";

static BLEServer* bleServer = nullptr;
static BLECharacteristic* alertCharacteristic = nullptr;

static bool bleClientConnected = false;
static bool bleInitialized = false;

// Forward declarations
static void printMemoryStatus(const char* stage);
static void sendBleJson(const char* message);
static void sendDeviceStatus();
static void sendSoundPrediction(int classIndex, float confidence);
static bool initializeBLE();

static bool microphoneSampling = false;
// Forward declarations for variables used by BLE functions
static bool monitoringEnabled = false;

extern const char* const classNames[20];

static void sendBleJson(const char* message) {
  if (!bleInitialized || alertCharacteristic == nullptr) {
    return;
  }

  alertCharacteristic->setValue(
      reinterpret_cast<const uint8_t*>(message),
      strlen(message)
  );

  if (bleClientConnected) {
    alertCharacteristic->notify();
    Serial.printf("[BLE TX] %s\n", message);
  }
}

// =====================================================
// BLE CONNECTION CALLBACKS
// =====================================================

class SoundSenseServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    bleClientConnected = true;

    Serial.println("[BLE] Flutter app connected.");
  }

  void onDisconnect(BLEServer* server) override {
    bleClientConnected = false;

    Serial.println("[BLE] Flutter app disconnected.");
    Serial.println("[BLE] Restarting advertising...");

    BLEDevice::startAdvertising();
  }
};

static void sendDeviceStatus() {
  char message[128];

  snprintf(
      message,
      sizeof(message),
      "{\"type\":\"status\","
      "\"esp32\":true,"
      "\"microphone\":%s,"
      "\"monitoring\":%s}",
      microphoneSampling ? "true" : "false",
      monitoringEnabled ? "true" : "false"
  );

  sendBleJson(message);
}

static void sendSoundPrediction(
    int classIndex,
    float confidence
) {
  if (classIndex < 0 || classIndex >= 20) {
    return;
  }

  char message[192];

  snprintf(
      message,
      sizeof(message),
      "{\"sound\":\"%s\","
      "\"confidence\":%.2f,"
      "\"timestamp\":%lu}",
      classNames[classIndex],
      confidence * 100.0f,
      static_cast<unsigned long>(millis())
  );

  sendBleJson(message);
}

// =====================================================
// BLE INITIALIZATION
// =====================================================

static bool initializeBLE() {
  printMemoryStatus("Before BLE Initialization");

  BLEDevice::init(BLE_DEVICE_NAME);

  bleServer = BLEDevice::createServer();

  if (bleServer == nullptr) {
    Serial.println("[BLE] ERROR: Server creation failed.");
    return false;
  }

  bleServer->setCallbacks(
      new SoundSenseServerCallbacks());

  BLEService* service =
      bleServer->createService(BLE_SERVICE_UUID);

  if (service == nullptr) {
    Serial.println("[BLE] ERROR: Service creation failed.");
    return false;
  }

  alertCharacteristic = service->createCharacteristic(
      BLE_CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ |
      BLECharacteristic::PROPERTY_NOTIFY);

  if (alertCharacteristic == nullptr) {
    Serial.println("[BLE] ERROR: Characteristic creation failed.");
    return false;
  }

  alertCharacteristic->addDescriptor(
      new BLE2902());

  alertCharacteristic->setValue(
      "{\"type\":\"status\",\"esp32\":true,"
      "\"microphone\":false,\"monitoring\":false}");

  service->start();

  BLEAdvertising* advertising =
      BLEDevice::getAdvertising();

  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->setScanResponse(true);

  BLEDevice::startAdvertising();

  bleInitialized = true;

  Serial.println("[BLE] Initialization successful.");
  Serial.println("[BLE] Advertising as SoundSense-ESP32.");

  printMemoryStatus("After BLE Initialization");

  return true;
}

// =====================================================
// 2. MODEL CONFIGURATION
// =====================================================

namespace {
constexpr size_t kTensorArenaSize = 104 * 1024;

uint8_t* tensor_arena = nullptr;

tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

bool model_ready = false;
}

static constexpr int SAMPLE_RATE = 16000;
static constexpr int AUDIO_SAMPLES = 32000;

static constexpr int FFT_SIZE = 512;
static constexpr int HOP_LENGTH = 160;
static constexpr int MEL_BANDS = 32;
static constexpr int MFCC_COUNT = 13;
static constexpr int FRAME_COUNT = 201;

static constexpr int AUDIO_CHUNK_SAMPLES = 2048;

static constexpr int AUDIO_CHUNK_COUNT =
    (AUDIO_SAMPLES + AUDIO_CHUNK_SAMPLES - 1) /
    AUDIO_CHUNK_SAMPLES;

static constexpr unsigned long DEBOUNCE_MS = 50;
static constexpr unsigned long ALERT_DURATION_MS = 5000;

I2SClass I2S;

static float hannWindow[FFT_SIZE];
static float melFilters[MEL_BANDS][FFT_SIZE / 2 + 1];
static float melEnergies[MEL_BANDS];
static float powerSpectrum[FFT_SIZE / 2 + 1];
static float realPart[FFT_SIZE];
static float imagPart[FFT_SIZE];

static int16_t* audioChunks[AUDIO_CHUNK_COUNT] = {};

static bool lastButtonReading = HIGH;
static bool stableButtonState = HIGH;

static unsigned long lastDebounceTime = 0;

// =====================================================
// 3. SOUND CLASSES
// =====================================================

const char* const classNames[20] = {
    "bell_sounds",
    "car_horn",
    "cat",
    "cheering",
    "clapping",
    "clock",
    "coughing",
    "crying_sobbing",
    "dog",
    "door_knock",
    "doorbell",
    "explosive_sounds",
    "footsteps",
    "general_alarm",
    "glass_breaking",
    "laughter",
    "phone_ringtone",
    "screaming",
    "siren",
    "transport_sounds"
};

// Sound categories
enum SoundCategory {
  SAFE_SOUND,
  MEDIUM_SOUND,
  DANGEROUS_SOUND
};

// Return an integer to avoid Arduino enum prototype issues.
static int classifySound(int classIndex) {
  switch (classIndex) {
    case 1:   // car_horn
    case 7:   // crying_sobbing
    case 11:  // explosive_sounds
    case 13:  // general_alarm
    case 14:  // glass_breaking
    case 17:  // screaming
    case 18:  // siren
      return DANGEROUS_SOUND;

    case 0:   // bell_sounds
    case 6:   // coughing
    case 8:   // dog
    case 9:   // door_knock
    case 10:  // doorbell
    case 16:  // phone_ringtone
    case 19:  // transport_sounds
      return MEDIUM_SOUND;

    default:
      return SAFE_SOUND;
  }
}

// =====================================================
// 4. RGB LED CONTROL
// Common-cathode LED: HIGH turns each channel ON.
// =====================================================

static void setLED(uint8_t red, uint8_t green, uint8_t blue) {
  analogWrite(LED_RED_PIN, red);
  analogWrite(LED_GREEN_PIN, green);
  analogWrite(LED_BLUE_PIN, blue);
}

static void ledOff() {
  setLED(0, 0, 0);
}

static void ledWhite() {
  setLED(255, 255, 255);
}

static void ledBlue() {
  setLED(0, 0, 255);
}

static void ledRed() {
  setLED(255, 0, 0);
}

static void ledYellow() {
  setLED(255, 255, 0);
}

static void ledGreen() {
  setLED(0, 255, 0);
}

// =====================================================
// 5. BUTTON CONTROL
// INPUT_PULLUP: pressed = LOW.
// =====================================================

static bool buttonPressed() {
  const bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonReading) {
    lastDebounceTime = millis();
    lastButtonReading = reading;
  }

  if (millis() - lastDebounceTime >= DEBOUNCE_MS) {
    if (reading != stableButtonState) {
      stableButtonState = reading;

      if (stableButtonState == LOW) {
        return true;
      }
    }
  }

  return false;
}

static void updateMonitoringState() {
  if (buttonPressed()) {
    monitoringEnabled = !monitoringEnabled;
    
    if (monitoringEnabled) {
      Serial.println("MONITORING ON");
      ledBlue();
    } else {
      Serial.println("MONITORING OFF");
      ledWhite();
    }
    sendDeviceStatus();
  }
}

// =====================================================
// 6. SOUND ALERT DISPLAY
// =====================================================

static void showSoundAlert(int classIndex) {
  const int category = classifySound(classIndex);

  switch (category) {
    case DANGEROUS_SOUND:
      Serial.println("ALERT: DANGEROUS SOUND");
      ledRed();
      break;

    case MEDIUM_SOUND:
      Serial.println("ALERT: MEDIUM-PRIORITY SOUND");
      ledYellow();
      break;

    case SAFE_SOUND:
      Serial.println("STATUS: SAFE SOUND");
      ledGreen();
      break;
  }

  const unsigned long alertStart = millis();

  // Keep the alert colour for 5 seconds.
  // Continue checking the button during the alert.
  while (millis() - alertStart < ALERT_DURATION_MS) {
    if (buttonPressed()) {
      monitoringEnabled = !monitoringEnabled;
      sendDeviceStatus();
      if (!monitoringEnabled) {
        Serial.println("MONITORING OFF");
        ledWhite();
        return;
      }

      Serial.println("MONITORING ON");
      ledBlue();
    }

    delay(5);
  }

  if (monitoringEnabled) {
    ledBlue();
  } else {
    ledWhite();
  }
}

// =====================================================
// 7. AUDIO BUFFER
// =====================================================

static bool allocateAudioChunks() {
  for (int chunk = 0; chunk < AUDIO_CHUNK_COUNT; ++chunk) {
    const int firstSample = chunk * AUDIO_CHUNK_SAMPLES;
    const int remaining = AUDIO_SAMPLES - firstSample;

    const int count =
        remaining < AUDIO_CHUNK_SAMPLES
            ? remaining
            : AUDIO_CHUNK_SAMPLES;

    audioChunks[chunk] = static_cast<int16_t*>(
        heap_caps_malloc(
            static_cast<size_t>(count) * sizeof(int16_t),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    if (audioChunks[chunk] == nullptr) {
      Serial.printf("ERROR: Audio chunk %d allocation failed.\n", chunk);

      for (int j = 0; j < chunk; ++j) {
        heap_caps_free(audioChunks[j]);
        audioChunks[j] = nullptr;
      }

      return false;
    }
  }

  return true;
}

static inline int16_t getAudioSample(int index) {
  return audioChunks[index / AUDIO_CHUNK_SAMPLES]
                    [index % AUDIO_CHUNK_SAMPLES];
}

static inline void setAudioSample(int index, int16_t sample) {
  audioChunks[index / AUDIO_CHUNK_SAMPLES]
             [index % AUDIO_CHUNK_SAMPLES] = sample;
}

// =====================================================
// 8. MFCC PREPROCESSING
// Preserved from your original V9-Lite sketch.
// =====================================================

static float hzToMel(float hz) {
  const float f_sp = 200.0f / 3.0f;

  if (hz < 1000.0f) {
    return hz / f_sp;
  }

  const float min_log_mel = 1000.0f / f_sp;
  const float logstep = logf(6.4f) / 27.0f;

  return min_log_mel + logf(hz / 1000.0f) / logstep;
}

static float melToHz(float mel) {
  const float f_sp = 200.0f / 3.0f;
  const float min_log_mel = 1000.0f / f_sp;
  const float logstep = logf(6.4f) / 27.0f;

  if (mel < min_log_mel) {
    return mel * f_sp;
  }

  return 1000.0f * expf(logstep * (mel - min_log_mel));
}

static void makeHann() {
  for (int n = 0; n < FFT_SIZE; ++n) {
    hannWindow[n] =
        0.5f - 0.5f * cosf(2.0f * PI * n / FFT_SIZE);
  }
}

static void makeMelFilters() {
  const int bins = FFT_SIZE / 2 + 1;

  const float melMin = hzToMel(0.0f);
  const float melMax = hzToMel(SAMPLE_RATE / 2.0f);

  float melPoints[MEL_BANDS + 2];
  float hzPoints[MEL_BANDS + 2];

  for (int i = 0; i < MEL_BANDS + 2; ++i) {
    melPoints[i] =
        melMin + (melMax - melMin) * i / (MEL_BANDS + 1);
  }

  for (int i = 0; i < MEL_BANDS + 2; ++i) {
    hzPoints[i] = melToHz(melPoints[i]);
  }

  for (int m = 0; m < MEL_BANDS; ++m) {
    const float left = hzPoints[m];
    const float center = hzPoints[m + 1];
    const float right = hzPoints[m + 2];

    const float enorm = 2.0f / (right - left);

    for (int k = 0; k < bins; ++k) {
      const float hz =
          static_cast<float>(k) * SAMPLE_RATE / FFT_SIZE;

      float weight = 0.0f;

      if (hz >= left && hz <= center && center > left) {
        weight = (hz - left) / (center - left);
      } else if (hz > center && hz <= right && right > center) {
        weight = (right - hz) / (right - center);
      }

      melFilters[m][k] = weight * enorm;
    }
  }
}

static void fft(float* re, float* im) {
  int j = 0;

  for (int i = 1; i < FFT_SIZE; ++i) {
    int bit = FFT_SIZE >> 1;

    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }

    j ^= bit;

    if (i < j) {
      float tmp = re[i];
      re[i] = re[j];
      re[j] = tmp;

      tmp = im[i];
      im[i] = im[j];
      im[j] = tmp;
    }
  }

  for (int len = 2; len <= FFT_SIZE; len <<= 1) {
    const float angle = -2.0f * PI / len;
    const float wlenR = cosf(angle);
    const float wlenI = sinf(angle);

    for (int i = 0; i < FFT_SIZE; i += len) {
      float wr = 1.0f;
      float wi = 0.0f;

      for (int k = 0; k < len / 2; ++k) {
        const int u = i + k;
        const int v = u + len / 2;

        const float vr = re[v] * wr - im[v] * wi;
        const float vi = re[v] * wi + im[v] * wr;

        re[v] = re[u] - vr;
        im[v] = im[u] - vi;

        re[u] += vr;
        im[u] += vi;

        const float nextWr = wr * wlenR - wi * wlenI;

        wi = wr * wlenI + wi * wlenR;
        wr = nextWr;
      }
    }
  }
}

static void computeFrame(int frameIndex) {
  const int frameStart =
      frameIndex * HOP_LENGTH - FFT_SIZE / 2;

  for (int n = 0; n < FFT_SIZE; ++n) {
    const int sampleIndex = frameStart + n;

    const float sample =
        (sampleIndex >= 0 && sampleIndex < AUDIO_SAMPLES)
            ? static_cast<float>(getAudioSample(sampleIndex)) / 32768.0f
            : 0.0f;

    realPart[n] = sample * hannWindow[n];
    imagPart[n] = 0.0f;
  }

  fft(realPart, imagPart);

  for (int k = 0; k <= FFT_SIZE / 2; ++k) {
    powerSpectrum[k] =
        realPart[k] * realPart[k] + imagPart[k] * imagPart[k];
  }

  for (int m = 0; m < MEL_BANDS; ++m) {
    float energy = 0.0f;

    for (int k = 0; k <= FFT_SIZE / 2; ++k) {
      energy += powerSpectrum[k] * melFilters[m][k];
    }

    melEnergies[m] = energy;
  }
}

static bool extractMFCC() {
  if (model_input == nullptr || model_input->data.f == nullptr) {
    Serial.println("ERROR: Model input tensor is unavailable.");
    return false;
  }

  constexpr float kMfccMean = -21.30274200439453f;
  constexpr float kMfccStd = 90.24063110351562f;

  float globalMaxDb = -INFINITY;

  // First pass: global maximum for top_db = 80.
  for (int frame = 0; frame < FRAME_COUNT; ++frame) {
    computeFrame(frame);

    for (int m = 0; m < MEL_BANDS; ++m) {
      const float db = 10.0f * log10f(
          fmaxf(melEnergies[m], 1.0e-10f));

      if (db > globalMaxDb) {
        globalMaxDb = db;
      }
    }
  }

  // Second pass: DCT and normalization.
  for (int frame = 0; frame < FRAME_COUNT; ++frame) {
    computeFrame(frame);

    float melDb[MEL_BANDS];

    for (int m = 0; m < MEL_BANDS; ++m) {
      float db = 10.0f * log10f(
          fmaxf(melEnergies[m], 1.0e-10f));

      if (db < globalMaxDb - 80.0f) {
        db = globalMaxDb - 80.0f;
      }

      melDb[m] = db;
    }

    for (int c = 0; c < MFCC_COUNT; ++c) {
      float sum = 0.0f;

      for (int m = 0; m < MEL_BANDS; ++m) {
        sum += melDb[m] *
            cosf(PI * c * (m + 0.5f) / MEL_BANDS);
      }

      const float scale =
          (c == 0)
              ? sqrtf(1.0f / MEL_BANDS)
              : sqrtf(2.0f / MEL_BANDS);

      const int index = frame * MFCC_COUNT + c;

      model_input->data.f[index] =
          (sum * scale - kMfccMean) / kMfccStd;
    }
  }

  return true;
}

// =====================================================
// 9. AUDIO CAPTURE
// =====================================================

static bool captureAudio() {
  for (int i = 0; i < AUDIO_SAMPLES; ++i) {
    int32_t raw = 0;

    const size_t got = I2S.readBytes(
        reinterpret_cast<char*>(&raw),
        sizeof(raw));

    if (got != sizeof(raw)) {
      Serial.printf(
          "ERROR: I2S read failed at sample %d (got %u bytes).\n",
          i,
          static_cast<unsigned>(got));

      return false;
    }

    // Preserved from your original working sketch.
    // Correctness depends on the I2S data alignment.
    setAudioSample(i, static_cast<int16_t>(raw >> 16));
  }

  int16_t minSample = INT16_MAX;
  int16_t maxSample = INT16_MIN;
  double sumSquares = 0.0;

  for (int i = 0; i < AUDIO_SAMPLES; ++i) {
    const int16_t sample = getAudioSample(i);

    if (sample < minSample) minSample = sample;
    if (sample > maxSample) maxSample = sample;

    sumSquares += static_cast<double>(sample) * sample;
  }

  const double rms = sqrt(sumSquares / AUDIO_SAMPLES);

  Serial.printf(
      "Audio min: %d | max: %d | RMS: %.2f\n",
      minSample, maxSample, rms);

  Serial.print("First 20 audio samples: ");

  for (int i = 0; i < 20; ++i) {
    Serial.printf("%d ", getAudioSample(i));
  }

  Serial.println();

  if (minSample == 0 && maxSample == 0) {
    Serial.println("WARNING: Audio samples are all zero.");
  }

  return true;
}

// =====================================================
// 10. MODEL INFERENCE
// =====================================================

static bool runInference() {
  if (interpreter == nullptr ||
      model_input == nullptr ||
      model_output == nullptr) {
    Serial.println("ERROR: Model tensors unavailable.");
    return false;
  }

  if (!extractMFCC()) {
    return false;
  }

  Serial.println("MFCC extraction complete.");
  Serial.println("Running V9-Lite inference...");

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Model inference failed.");
    return false;
  }

  int topIndex = -1;
  float topScore = -INFINITY;

  Serial.println("========== ALL MODEL SCORES ==========");

  for (int i = 0; i < 20; ++i) {
    const float score = model_output->data.f[i];

    Serial.printf("Class %02d (%s): %.6f\n",
                  i, classNames[i], score);

    if (score > topScore) {
      topScore = score;
      topIndex = i;
    }
  }

  if (topIndex >= 0) {
    Serial.println("========== TOP PREDICTION ==========");

    Serial.printf(
        "%s: %.6f (%.2f%%)\n",
        classNames[topIndex],
        topScore,
        topScore * 100.0f
      );
    sendSoundPrediction(topIndex, topScore);
    showSoundAlert(topIndex);
  }

  return true;
}

// =====================================================
// MEMORY DIAGNOSTICS
// =====================================================

static void printMemoryStatus(const char* stage) {
  Serial.printf("\n--- MEMORY: %s ---\n", stage);

  Serial.printf(
      "Free internal heap: %u bytes\n",
      static_cast<unsigned>(
          heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));

  Serial.printf(
      "Largest internal block: %u bytes\n",
      static_cast<unsigned>(
          heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)));

  Serial.println("-------------------------");
}

// =====================================================
// 11. SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_BLUE_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  ledOff();

  Serial.println();
  Serial.println("======================================");
  Serial.println("SOUNDSENSE V9-LITE");
  Serial.println("ESP32-WROOM-32 + INMP441");
  Serial.println("======================================");

  Serial.println("Initializing I2S microphone...");

  I2S.setPins(
      I2S_BCLK_PIN,
      I2S_WS_PIN,
      -1,
      I2S_SD_PIN,
      -1);

  if (!I2S.begin(
          I2S_MODE_STD,
          SAMPLE_RATE,
          I2S_DATA_BIT_WIDTH_32BIT,
          I2S_SLOT_MODE_MONO,
          I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR: I2S initialization failed.");
    ledOff();

    while (true) {
      delay(1000);
    }
  }

  Serial.println("I2S initialized.");

  makeHann();
  makeMelFilters();

  Serial.printf(
      "Model file size: %u bytes\n",
      static_cast<unsigned>(soundsense_v9_lite_shifted_bn_model_len));

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(
          kTensorArenaSize,
          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

  if (tensor_arena == nullptr) {
    Serial.println("ERROR: Tensor arena allocation failed.");
    ledOff();
    return;
  }

  model = tflite::GetModel(soundsense_v9_lite_shifted_bn_model);

  if (model == nullptr ||
      model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model/schema version problem.");
    ledOff();
    return;
  }

  static tflite::MicroMutableOpResolver<7> resolver(
      &micro_error_reporter);

  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMul() != kTfLiteOk ||
      resolver.AddAdd() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("ERROR: Operator registration failed.");
    ledOff();
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      &micro_error_reporter);

  interpreter = &static_interpreter;
  printMemoryStatus("Before Tensor Arena");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    ledOff();
    return;
  }
  printMemoryStatus("After Tensor Arena");

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  if (model_input == nullptr ||
      model_output == nullptr ||
      model_input->type != kTfLiteFloat32 ||
      model_output->type != kTfLiteFloat32 ||
      model_input->dims->size != 4 ||
      model_input->dims->data[0] != 1 ||
      model_input->dims->data[1] != 201 ||
      model_input->dims->data[2] != 13 ||
      model_input->dims->data[3] != 1 ||
      model_output->dims->size != 2 ||
      model_output->dims->data[0] != 1 ||
      model_output->dims->data[1] != 20) {
    Serial.println("ERROR: Unexpected model input/output shape.");
    ledOff();
    return;
  }

  Serial.printf(
      "Tensor arena used: %u bytes\n",
      static_cast<unsigned>(interpreter->arena_used_bytes()));

  if (!allocateAudioChunks()) {
    Serial.println("ERROR: Audio buffer allocation failed.");
    ledOff();
    return;
  }
  printMemoryStatus("After Audio Buffers");

  model_ready = true;

  monitoringEnabled = false;
  microphoneSampling = false;

  ledWhite();

  Serial.println("Model initialization successful.");

  // Initialize BLE after allocating the model and audio buffers.
  if (initializeBLE()) {
    sendDeviceStatus();
  } else {
    Serial.println(
      "[BLE] Initialization failed. "
      "Model and microphone may still work without BLE."
    );
  }

  printMemoryStatus("After BLE Initialization Attempt");

  Serial.println("Monitoring OFF: LED WHITE.");
  Serial.println("Press button on GPIO 27 to toggle monitoring.");
}

// =====================================================
// 12. MAIN LOOP
// =====================================================

void loop() {
  if (!model_ready) {
    ledOff();
    delay(500);
    return;
  }

  updateMonitoringState();

  if (!monitoringEnabled) {
    ledWhite();
    delay(5);
    return;
  }

  ledBlue();

  Serial.println();
  Serial.println("Capturing 2 seconds of audio...");

  microphoneSampling = true;
  sendDeviceStatus();

  const bool audioCaptured = captureAudio();

  microphoneSampling = false;
  sendDeviceStatus();

  if (!audioCaptured) {
    Serial.println("[MIC] Audio capture failed.");

    ledBlue();
    delay(100);
    return;
  }

// Run model inference and transmit its prediction.
runInference();

  if (monitoringEnabled) {
    ledBlue();
  } else {
    ledWhite();
  }

  delay(10);
}
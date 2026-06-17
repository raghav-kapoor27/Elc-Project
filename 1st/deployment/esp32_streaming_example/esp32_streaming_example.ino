#include <Arduino.h>
#include <WiFi.h>

// Single-lead Lead I front-end analog output connected to ADC_PIN.
// Stream samples to a host running scripts/infer_stream.py or a socket bridge.

constexpr int ADC_PIN = 34;
constexpr int SAMPLE_RATE_HZ = 250;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

const char* WIFI_SSID = "";
const char* WIFI_PASSWORD = "";
const char* HOST = "192.168.1.10";
constexpr uint16_t PORT = 5000;

WiFiClient client;
uint32_t next_sample_us = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  if (strlen(WIFI_SSID) > 0) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
      delay(250);
    }
    client.connect(HOST, PORT);
  }
  next_sample_us = micros();
}

void loop() {
  const uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_PERIOD_US;
    const int raw = analogRead(ADC_PIN);
    const float volts = (raw / 4095.0f) * 3.3f;

    Serial.println(volts, 6);
    if (client.connected()) {
      client.println(volts, 6);
    }
  }
}

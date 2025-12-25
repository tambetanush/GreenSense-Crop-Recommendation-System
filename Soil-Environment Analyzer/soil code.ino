#include <Adafruit_BMP280.h>
#include <Adafruit_Sensor.h>
#include <BH1750.h>
#include <HTTPClient.h>
#include <ModbusMaster.h>
#include <WiFi.h>
#include <Wire.h>
#include "DHT.h"

// ---------------- WIFI ----------------
const char* ssid = "1Plus";
const char* password = "87654321";
const char* scriptURL =
    "https://script.google.com/macros/s/"
    "AKfycbyTpysfpeQIB3wNvym2Gk8cx_dPQtLW0cB48RO07K9LpPoWe2hl_iRPpjvWdeVWgmk/"
    "exec";

// ---------------- SENSOR DATA STRUCT ----------------
struct SensorData {
    float bmpTemp = 0, bmpPress = 0, bmpAlt = 0;
    float light = 0;
    float dhtTemp = 0, dhtHum = 0;
    float envTemp = 0;
    float co2 = 0, co = 0, nh3 = 0;
    float soilM = 0, soilT = 0, ph = 0, N = 0, P = 0, K = 0;
};

// ---------------- BMP280 ----------------
#define BMP_SDA 41
#define BMP_SCL 40
Adafruit_BMP280 bmp;

// ---------------- BH1750 ----------------
#define BH_SDA 41
#define BH_SCL 40
BH1750 lightMeter;

// ---------------- DHT11 ----------------
#define DHTPIN 6
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ---------------- MQ135 ----------------
#define MQ135_PIN 7
#define ADC_BITS 12
#define ADC_MAX 4095.0
#define VCC 3.3
#define CLEAN_AIR_FACTOR 3.6
#define RLOAD 10.0
float R0 = 0;

float getRS() {
    int adc = analogRead(MQ135_PIN);
    float Vout = adc * (VCC / ADC_MAX);
    if (Vout < 0.01) Vout = 0.01;
    if (Vout > (VCC - 0.01)) Vout = VCC - 0.01;
    return (VCC - Vout) * RLOAD / Vout;
}

// ---------------- ZTS-3001 ----------------
#define RXD2 16
#define TXD2 17
ModbusMaster node;

// ---------------- Aggregation ----------------
SensorData avg;
int sampleCount = 0;

// ---------------- Setup ----------------
void setup() {
    Serial.begin(115200);
    analogReadResolution(ADC_BITS);
    Serial.println("\nInitializing sensors...");

    // Wi-Fi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi.");

    // BMP280
    Wire.begin(BMP_SDA, BMP_SCL);
    if (bmp.begin(0x76))
        Serial.println("BMP280 detected");
    else
        Serial.println("BMP280 not found");

    // BH1750
    Wire.begin(BH_SDA, BH_SCL);
    lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);

    // DHT
    dht.begin();

    // MQ135 calibration
    float RS_sum = 0;
    for (int i = 0; i < 50; i++) {
        RS_sum += getRS();
        delay(50);
    }
    R0 = (RS_sum / 50.0) / CLEAN_AIR_FACTOR;
    Serial.printf("MQ135 Calibrated R0=%.2f\n", R0);

    // ZTS sensor
    Serial2.begin(4800, SERIAL_8N1, RXD2, TXD2);
    node.begin(1, Serial2);

    configTime(19800, 0, "pool.ntp.org");  // IST timezone

    Serial.println("=== Setup complete ===\n");
}

// ---------------- Read all sensors ----------------
SensorData readSensors() {
    SensorData d;

    d.bmpTemp = bmp.readTemperature();
    d.bmpPress = bmp.readPressure() / 100.0;
    d.bmpAlt = bmp.readAltitude(1013.25);

    d.light = lightMeter.readLightLevel();

    delay(200);

    d.dhtHum = dht.readHumidity();
    d.dhtTemp = dht.readTemperature();

    if (isnan(d.dhtTemp)) d.dhtTemp = d.bmpTemp;
    if (isnan(d.dhtHum)) d.dhtHum = 0;

    d.envTemp = (d.bmpTemp + d.dhtTemp) / 2.0;

    float RS = getRS();
    if (RS <= 0) RS = 0.01;
    float ratio = RS / R0;
    d.co2 = 110.47 * pow(ratio, -2.862);
    d.co = 605.18 * pow(ratio, -3.937);
    d.nh3 = 77.255 * pow(ratio, -3.18);

    uint8_t result = node.readHoldingRegisters(0x0000, 6);
    if (result == node.ku8MBSuccess) {
        d.soilM = node.getResponseBuffer(0) / 10.0f;
        d.soilT = node.getResponseBuffer(1) / 10.0f;
        d.ph = node.getResponseBuffer(2) / 10.0f;
        d.N = node.getResponseBuffer(3);
        d.P = node.getResponseBuffer(4);
        d.K = node.getResponseBuffer(5);
    }
    return d;
}

// ---------------- Average and Upload ----------------
void uploadAverages(SensorData d) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi not connected. Skipping upload.");
        return;
    }

    HTTPClient http;
    http.begin(scriptURL);
    http.addHeader("Content-Type", "application/json");
    String t = String(year()) + "-" + String(month()) + "-" + String(day()) +
               " " + String(hour()) + ":" + String(minute()) + ":" +
               String(second());

    String payload = "{";
    payload += "\"timestamp\":\"" + t + "\",";
    payload += "\"ID\":\"1" + "\",";
    payload += "\"Env_Temp\":" + String(d.envTemp, 2) + ",";
    // payload += "\"BMP_Temp\":" + String(d.bmpTemp,2) + ",";
    payload += "\"BMP_Pressure\":" + String(d.bmpPress, 2) + ",";
    payload += "\"BMP_Altitude\":" + String(d.bmpAlt, 2) + ",";
    payload += "\"Light\":" + String(d.light, 2) + ",";
    // payload += "\"DHT_Temp\":" + String(d.dhtTemp,2) + ",";
    payload += "\"DHT_Humidity\":" + String(d.dhtHum, 2) + ",";
    payload += "\"MQ135_CO2\":" + String(d.co2, 2) + ",";
    payload += "\"MQ135_CO\":" + String(d.co, 2) + ",";
    payload += "\"MQ135_NH3\":" + String(d.nh3, 2) + ",";
    payload += "\"Soil_Moisture\":" + String(d.soilM, 2) + ",";
    payload += "\"Soil_Temp\":" + String(d.soilT, 2) + ",";
    payload += "\"pH\":" + String(d.ph, 2) + ",";
    payload += "\"N\":" + String(d.N, 2) + ",";
    payload += "\"P\":" + String(d.P, 2) + ",";
    payload += "\"K\":" + String(d.K, 2);
    payload += "}";

    int code = http.POST(payload);
    Serial.printf("Upload → Code %d\n", code);
    http.end();
}

// ---------------- Loop ----------------
void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long lastAttempt = 0;
        if (millis() - lastAttempt > 10000) {  // Reconnect every 10 sec
            WiFi.disconnect();
            WiFi.begin(ssid, password);
            lastAttempt = millis();
            Serial.println("Reconnecting WiFi...");
        }
    }

    SensorData now = readSensors();

    avg.bmpTemp += now.bmpTemp;
    avg.bmpPress += now.bmpPress;
    avg.bmpAlt += now.bmpAlt;
    avg.light += now.light;
    avg.dhtTemp += now.dhtTemp;
    avg.dhtHum += now.dhtHum;
    avg.envTemp += now.envTemp;
    avg.co2 += now.co2;
    avg.co += now.co;
    avg.nh3 += now.nh3;
    avg.soilM += now.soilM;
    avg.soilT += now.soilT;
    avg.ph += now.ph;
    avg.N += now.N;
    avg.P += now.P;
    avg.K += now.K;

    sampleCount++;
    Serial.printf("Sample %d/6 collected\n", sampleCount);

    if (sampleCount >= 6) {
        avg.bmpTemp /= sampleCount;
        avg.bmpPress /= sampleCount;
        avg.bmpAlt /= sampleCount;
        avg.light /= sampleCount;
        avg.dhtTemp /= sampleCount;
        avg.dhtHum /= sampleCount;
        avg.envTemp /= sampleCount;
        avg.co2 /= sampleCount;
        avg.co /= sampleCount;
        avg.nh3 /= sampleCount;
        avg.soilM /= sampleCount;
        avg.soilT /= sampleCount;
        avg.ph /= sampleCount;
        avg.N /= sampleCount;
        avg.P /= sampleCount;
        avg.K /= sampleCount;

        Serial.println("Uploading averaged data...");
        uploadAverages(avg);

        avg = SensorData();
        sampleCount = 0;
    }

    delay(10000);  // 10 seconds per sample
}

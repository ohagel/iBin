#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define PIN 32
#define NUMPIXELS 28

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  // put your setup code here, to run once:
  pixels.begin();
  Serial.begin(115200);
}

const int packetSize = 6; // 1byte get/set + 1byte address + 4byte data
byte recvData[packetSize];

int32_t values[3]; //hallsensor, weight, led

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    for (int i = 0; i < packetSize; i++) {
      recvData[i] = Serial.read();
    }
    
    switch (recvData[1])
    {
    case 0:
      for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, pixels.Color(255, 0, 0));
      }
      pixels.show();
      break;
    case 1:
      for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, pixels.Color(0, 255, 0));
      }
      pixels.show();
      break;
    case 2:
      for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, pixels.Color(0, 0, 255));
      }
      pixels.show();
      break;

    default:
      break;
    }
  }
}
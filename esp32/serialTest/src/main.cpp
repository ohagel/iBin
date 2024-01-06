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

const int packetSize = 6; // 1byte get(0)/set(1) + 1byte address + 4byte data
byte recvData[packetSize];

int32_t values[] = {10000,0,0}; //hallsensor(0), weight(1), led(2)

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    for (int i = 0; i < packetSize; i++) {
      recvData[i] = Serial.read();
    }
    
    int32_t packedData = (recvData[2] << 24) | (recvData[3] << 16) | (recvData[4] << 8) | recvData[5];

    switch (recvData[1])
    {
    case 0:
      if (!recvData[0]) {
        byte buf[] = {0, 0, values[0]>>24, values[0]>>16, values[0]>>8, values[0]};
        Serial.write(buf, 6);
      }
      break;
    case 1:
      
      break;
    case 2:
      if (recvData[0]) {
        values[2] = packedData;
        for (int i = 0; i < NUMPIXELS; i++) {
          pixels.setPixelColor(i, pixels.Color((int)values[2], (int)values[2], (int)values[2]));
        }
        pixels.show();
      }
      break;

    default:
      break;
    }
  }
}
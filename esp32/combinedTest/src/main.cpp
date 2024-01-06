#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include "HX711.h"

#define PIN 32
#define NUMPIXELS 28

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

#define HALL_PIN 27
const int LOADCELL_DOUT_PIN = 5;
const int LOADCELL_SCK_PIN = 4;

HX711 scale;

void setup() {
  pinMode(HALL_PIN, INPUT_PULLUP);
  pixels.begin();
  for(int i=0; i<NUMPIXELS; i++){
    pixels.setPixelColor(i, pixels.Color(255,255,255));
    pixels.show();
  }
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  Serial.begin(115200);
}

void loop() {
  int hallVal = digitalRead(HALL_PIN);
  //int scaleVal = scale.read_average(5);
  for(int i=0; i<NUMPIXELS; i++){
    pixels.setPixelColor(i, pixels.Color(!hallVal*255,!hallVal*255,!hallVal*255));
    pixels.show();
  }
  //Serial.print(hallVal);
  //Serial.print("\t");
  //Serial.println(scaleVal);
}
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define PIN 32
#define NUMPIXELS 28

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);




void setup() {
  //init with blue color
  pixels.begin();
  //set all to blue in for loop
  for(int i=0; i<NUMPIXELS; i++){
    pixels.setPixelColor(i, pixels.Color(255,0,0));
    pixels.show();
    
  }
}

void loop() {
  // put your main code here, to run repeatedly:
}


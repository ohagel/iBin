#include <Arduino.h>

#define HALL_PIN 27

void setup() {
  //setup as input with pullup
  //pinMode(HALL_PIN, INPUT_PULLUP);
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(analogRead(HALL_PIN));
}
#include <Arduino.h>

#include "HX711.h"

// HX711 circuit wiring
const int LOADCELL_DOUT_PIN = 5;
const int LOADCELL_SCK_PIN = 4;

HX711 scale;

void setup() {
  Serial.begin(115200);
  
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
}

void loop() {

  Serial.print("read average: \t\t");
  Serial.println(scale.read_average(5)); 
  
}

//calibration factor will be the (reading)/(known weight)


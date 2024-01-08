#include <Arduino.h>

#include "HX711.h"

// HX711 circuit wiring
const int LOADCELL_DOUT_PIN = 5;
const int LOADCELL_SCK_PIN = 4;

HX711 scale;

void setup() {
  Serial.begin(115200);
  
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  float calibrationfactor = (62860)/(170);
  scale.set_scale(calibrationfactor);
  scale.tare(); 
  

}

void loop() {

  Serial.println("read average: \t\t");
  Serial.print("raw: \t\t\t\t");
  Serial.println(scale.read_average(5)); //raw
  Serial.print("calibrated: \t\t\t");
  Serial.println(scale.get_units(5), 1); //calibrated
  
}

//calibration factor will be the (reading)/(known weight)


int noiseInputPin = 6;
int toneInputPin = 7;
int soundOutputPin = 8;
int soundDuration = 200; // ms
int soundStartTime = 0;
int toneFreq = 8000; // Hz
int toneHalfPeriod = int(1e6/toneFreq/2); // microseconds

void setup() {
  pinMode(noiseInputPin,INPUT);
  pinMode(toneInputPin,INPUT);
  pinMode(soundOutputPin,OUTPUT);
}

void loop() {
  if (digitalRead(toneInputPin) == HIGH) {
    tone(soundOutputPin,toneFreq,soundDuration);
    delay(soundDuration+10);
//    soundStartTime = millis();
//    while (millis()-soundStartTime < soundDuration) {
//      digitalWrite(soundOutputPin,HIGH);
//      delayMicroseconds(toneHalfPeriod);
//      digitalWrite(soundOutputPin,LOW);
//      delayMicroseconds(toneHalfPeriod);
//    }
  }
  else if (digitalRead(noiseInputPin) == HIGH) {
    soundStartTime = millis();
    while (millis()-soundStartTime < soundDuration) {
      digitalWrite(soundOutputPin,random(2));
    }
    digitalWrite(soundOutputPin,LOW);
  }
}

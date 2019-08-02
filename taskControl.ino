int noiseInputPin = 6;
int toneInputPin = 7;
int soundOutputPin = 9;
int triggerOutputPin = 2;
int soundStartTime = 0;
int soundDuration = 200; // ms
int toneFreq = 8000; // Hz

void setup() {
  pinMode(noiseInputPin,INPUT);
  pinMode(toneInputPin,INPUT);
  pinMode(soundOutputPin,OUTPUT);
  pinMode(triggerOutputPin,OUTPUT);
}

void loop() {
  if (digitalRead(toneInputPin) == HIGH) {
    tone(soundOutputPin,toneFreq,soundDuration);
  }
  else if (digitalRead(noiseInputPin) == HIGH) {
//    soundStartTime = millis();
//    while (millis()-soundStartTime < soundDuration) {
//      digitalWrite(soundOutputPin,random(2));
//      delayMicroseconds(50);
//    }
//    digitalWrite(soundOutputPin,LOW);
    digitalWrite(triggerOutputPin,HIGH);
    delay(soundDuration);
    digitalWrite(triggerOutputPin,LOW);
  }
}

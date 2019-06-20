int noiseInputPin = 6;
int toneInputPin = 7;
int soundOutputPin = 8;
int soundDuration = 200;
int t = 0;

void setup() {
  pinMode(noiseInputPin, INPUT);
  pinMode(toneInputPin, INPUT);
  pinMode(soundOutputPin, OUTPUT);
}

void loop() {
  if (digitalRead(toneInputPin) == HIGH) {
    tone(soundOutputPin,8000,soundDuration);
    delay(1000);
  }
  else if (digitalRead(noiseInputPin) == HIGH) {
    t = millis();
    while (millis()-t < soundDuration) {
      if (random(0,2)) {
        digitalWrite(soundOutputPin,HIGH);
      }
      else {
        digitalWrite(soundOutputPin,LOW);
      }
    }
    digitalWrite(soundOutputPin,LOW);
    delay(1000);
  }
}

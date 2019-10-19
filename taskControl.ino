int noiseInputPin = 6;
int toneInputPin = 7;
int soundOutputPin = 9;
int toneDuration = 200; // ms
int toneFreq = 8000;
int noiseDuration = 1000;
unsigned long noiseStartTime = 0;

void setup() {
  pinMode(noiseInputPin,INPUT);
  pinMode(toneInputPin,INPUT);
  pinMode(soundOutputPin,OUTPUT);
}

void loop() {
  if (digitalRead(toneInputPin) == HIGH) {
    tone(soundOutputPin,toneFreq,toneDuration);
  }
  else if (digitalRead(noiseInputPin) == HIGH) {
    noTone(soundOutputPin);
    noiseStartTime = millis();
    while (millis()-noiseStartTime < noiseDuration) {
      digitalWrite(soundOutputPin,random(2));
    }
    digitalWrite(soundOutputPin,LOW);
  }
}

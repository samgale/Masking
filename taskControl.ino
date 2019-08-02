int noiseInputPin = 6;
int toneInputPin = 7;
int soundOutputPin = 9;
unsigned long soundStartTime = 0;
int soundDuration = 200; // ms
int toneFreq = 8000;

void setup() {
  pinMode(noiseInputPin,INPUT);
  pinMode(toneInputPin,INPUT);
  pinMode(soundOutputPin,OUTPUT);
}

void loop() {
  if (digitalRead(toneInputPin) == HIGH) {
    tone(soundOutputPin,toneFreq,soundDuration);
  }
  else if (digitalRead(noiseInputPin) == HIGH) {
    noTone(soundOutputPin);
    soundStartTime = millis();
    while (millis()-soundStartTime < soundDuration) {
      digitalWrite(soundOutputPin,random(2));
    }
    digitalWrite(soundOutputPin,LOW);
  }
}

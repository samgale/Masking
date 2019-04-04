int rewardInputPin = 7;
int solenoidOutputPin = 8;
int lastInput = 0;
unsigned long rewardDuration = 75;
unsigned long rewardRefractoryPeriod = 1000;

void setup() {
  pinMode(rewardInputPin, INPUT);
  pinMode(solenoidOutputPin, OUTPUT);
  digitalWrite(solenoidOutputPin, HIGH);
}

void loop() {
  if (digitalRead(rewardInputPin) == HIGH) {
    if (lastInput == 0) {
      digitalWrite(solenoidOutputPin, LOW);
      delay(rewardDuration);
      digitalWrite(solenoidOutputPin, HIGH);
      delay(rewardRefractoryPeriod);
    }
    lastInput = 1;
  }
  else {
    lastInput = 0;
  }
}

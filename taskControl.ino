int rewardInputPin = 7;
int solenoidOutputPin = 8;
unsigned long rewardDuration = 100;
unsigned long rewardRefractoryPeriod = 1000;

void setup() {
  pinMode(rewardInputPin, INPUT);
  pinMode(solenoidOutputPin, OUTPUT);
  digitalWrite(solenoidOutputPin, HIGH);
}

void loop() {
  if (digitalRead(rewardInputPin) == HIGH) {
    digitalWrite(solenoidOutputPin, LOW);
    delay(rewardDuration);
    digitalWrite(solenoidOutputPin, HIGH);
    delay(rewardRefractoryPeriod);
  }
}
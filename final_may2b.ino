#include <Servo.h>

const int stepPin = 2; // Change step pin to 2 for A4988 driver
const int dirPin = 4;
const int stepsPerRevolution = 3200; // Steps per revolution for NEMA 17

Servo myservo; // create servo object to control a servo
int pos = 0; 

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  myservo.attach(8); // attaches the servo on pin 8 to the servo object
  Serial.begin(9600); // initialize serial communication
  while (!Serial) {}  // Wait for serial port to connect
}

void loop() {
  // Wait for signal from Python script
  while (!Serial.available()) {}
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    Serial.println("Received: " + input); // Debug print
    if (input == "start") {
      rotateServo();
      rotateStepper();
      delay(1000); // Adjust this delay according to the time it takes for stepper rotation
      rotateServo2();
      Serial.println("done"); // Signal Python script that rotation is done
    } else if (input == "start2") {
      rotateStepper();
      delay(1000); // Adjust this delay according to the time it takes for stepper rotation
      Serial.println("done"); // Signal Python script that rotation is done
    }
    // Flush serial buffer
    while (Serial.available() > 0) {
      char c = Serial.read();
    }
  }
}

void rotateServo() {
  // Rotate from 0 to 180 degrees
  for (pos = 0; pos <= 220; pos += 1) {
    myservo.write(pos); // tell servo to go to position in variable 'pos'
    delay(5); // waits 5 ms for the servo to reach the position
  }
  // delay(5000); // wait for 1 second at the end position

  // // Rotate from 180 to 0 degrees
  // for (pos = 220; pos >= 0; pos -= 1) {
  //   myservo.write(pos); // tell servo to go to position in variable 'pos'
  //   delay(5); // waits 5 ms for the servo to reach the position
  // }
  delay(1000); // wait for 1 second at the end position
  Serial.println("done"); // Signal Python script that rotation is done
}

void rotateStepper() {
  Serial.println("Rotating stepper..."); // Debug print
  digitalWrite(dirPin, HIGH); // Set direction to forward (clockwise)
  
  // Steps for one revolution
  int steps = stepsPerRevolution;
  
  // Step delay for suitable speed (lower values for faster speed)
  // int stepDelay = 500; // Adjust as necessary
  
  for (int x = 0; x < steps; x++) {
    digitalWrite(stepPin, HIGH); // Send step pulse
    delayMicroseconds(4000); // Minimum delay for the step pulse (do not reduce this too much)
    digitalWrite(stepPin, LOW); // Complete step pulse
    delayMicroseconds(4000); // Adjust this value as per your motor and driver
  }
  
  delay(1000); // A small delay between rotations
  
}


void rotateServo2() {
  // // Rotate from 0 to 180 degrees
  // for (pos = 0; pos <= 220; pos += 1) {
  //   myservo.write(pos); // tell servo to go to position in variable 'pos'
  //   delay(5); // waits 5 ms for the servo to reach the position
  // }
  // delay(5000); // wait for 1 second at the end position

  // Rotate from 180 to 0 degrees
  for (pos = 220; pos >= 0; pos -= 1) {
    myservo.write(pos); // tell servo to go to position in variable 'pos'
    delay(5); // waits 5 ms for the servo to reach the position
  }
  delay(1000); // wait for 1 second at the end position
  Serial.println("done"); // Signal Python script that rotation is done
}


/*
  Legacy Input Companion LED
  --------------------------
  This sketch makes the built-in LED (pin 13) reflect the current stage of the
  performance.  When the glove prompt appears, plug the Arduino in, then send
  one of the commands below via the Serial monitor (9600 baud, newline-delimited).

  Commands:
    STAGE:INIT
      -> steady LED (init dialog stage)
    STAGE:CALIBRATE:STEP=N
      -> blinking LED where the blink rate (and pattern) changes with the
         calibration step number 1..5+ (a higher step makes the blink faster
         and a brief double-flash to show growing anxiety).
    STAGE:FINISH
      -> slow steady pulse to signal completion and calm acceptance.

  The wires on the glove can remain unconnected—this is just a visual cue.
*/

const int ledPin = LED_BUILTIN;

enum StageState {
  STAGE_IDLE,
  STAGE_INIT,
  STAGE_CALIBRATION,
  STAGE_FINISH
};

StageState currentStage = STAGE_IDLE;
unsigned long blinkInterval = 500;
unsigned long lastToggle = 0;
bool ledModeOn = false;
int calibrationStep = 0;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("Legacy Glove LED ready. Awaiting stage commands.");
  digitalWrite(ledPin, LOW);
}

void loop() {
  handleSerial();
  updateLed();
}

void handleSerial() {
  if (Serial.available() == 0) {
    return;
  }
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }
  if (line.equalsIgnoreCase("STAGE:INIT")) {
    enterInitStage();
  } else if (line.startsWith("STAGE:CALIBRATE")) {
    enterCalibrationStage(line);
  } else if (line.equalsIgnoreCase("STAGE:FINISH")) {
    enterFinishStage();
  }
}

void enterInitStage() {
  currentStage = STAGE_INIT;
  ledModeOn = true;
  digitalWrite(ledPin, HIGH);
  Serial.println("Stage locked to INIT (steady LED).");
}

void enterCalibrationStage(const String &command) {
  currentStage = STAGE_CALIBRATION;
  calibrationStep = extractStep(command);
  calibrationStep = constrain(calibrationStep, 1, 8);
  blinkInterval = 600 - calibrationStep * 50;
  if (blinkInterval < 100) {
    blinkInterval = 100;
  }
  ledModeOn = false;
  digitalWrite(ledPin, LOW);
  Serial.print("Calibration stage ");
  Serial.print(calibrationStep);
  Serial.print(" -> blink interval ");
  Serial.print(blinkInterval);
  Serial.println(" ms.");
}

void enterFinishStage() {
  currentStage = STAGE_FINISH;
  blinkInterval = 900;
  ledModeOn = false;
  digitalWrite(ledPin, LOW);
  Serial.println("Finish stage: gentle pulse.");
}

int extractStep(const String &command) {
  int colon = command.indexOf(':', 14); // after "STAGE:CALIBRATE"
  if (colon < 0) {
    return 1;
  }
  String stepValue = command.substring(colon + 1);
  stepValue.trim();
  return stepValue.toInt();
}

void updateLed() {
  unsigned long now = millis();
  switch (currentStage) {
    case STAGE_INIT:
      // LED already steady ON in this stage.
      break;
    case STAGE_CALIBRATION:
      if (now - lastToggle >= blinkInterval) {
        ledModeOn = !ledModeOn;
        digitalWrite(ledPin, ledModeOn ? HIGH : LOW);
        lastToggle = now;
        if (calibrationStep >= 5 && ledModeOn) {
          // quick double-tap for higher steps.
          blinkInterval = max(120UL, blinkInterval - 30);
        }
      }
      break;
    case STAGE_FINISH:
      if (now - lastToggle >= blinkInterval) {
        ledModeOn = !ledModeOn;
        digitalWrite(ledPin, ledModeOn ? HIGH : LOW);
        lastToggle = now;
      }
      break;
    default:
      // Keep LED off until a command arrives.
      digitalWrite(ledPin, LOW);
      break;
  }
}

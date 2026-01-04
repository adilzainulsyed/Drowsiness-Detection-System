#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int buttonPin = 7;
bool drowsy = true;          
int lastButtonState = HIGH;
const int buzzer = 9; 
const int motorPin = 10;

void setup() {
  lcd.begin(16, 2);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzer, OUTPUT);
  pinMode(motorPin, OUTPUT);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Normal");       
}

void loop() {
  int currentButtonState = digitalRead(buttonPin);

  if (lastButtonState == HIGH && currentButtonState == LOW) {
    drowsy = !drowsy;        

    lcd.clear();
    lcd.setCursor(0, 0);

    if (drowsy){
      lcd.print("Drowsy!");
      tone(buzzer, 100);
      digitalWrite(motorPin, HIGH);
	  delay(500); 
    }
    else{
      lcd.print("Normal");
      noTone(buzzer);
      digitalWrite(motorPin, LOW);
	  delay(500);
  }

    delay(200);              
  }

  lastButtonState = currentButtonState;
}

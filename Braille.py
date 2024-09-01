import RPi.GPIO as GPIO
import time

# Define the GPIO pins corresponding to each Braille dot
braille_pins = {
    1: 17,  # Dot 1
    2: 18,  # Dot 2
    3: 27,  # Dot 3
    4: 22,  # Dot 4
    5: 23,  # Dot 5
    6: 24,  # Dot 6
}

# Initialize the GPIO pins
GPIO.setmode(GPIO.BCM)
for pin in braille_pins.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Define Braille patterns for each letter (A-Z)
braille_mapping = {
    'A': [1],        # A: 100000
    'B': [1, 2],     # B: 101000
    'C': [1, 4],     # C: 110000
    'D': [1, 4, 5],  # D: 110100
    'E': [1, 5],     # E: 100100
    'F': [1, 2, 4],  # F: 111000
    'G': [1, 2, 4, 5], # G: 111100
    'H': [1, 2, 5],  # H: 101100
    'I': [2, 4],     # I: 011000
    'J': [2, 4, 5],  # J: 011100
    'K': [1, 3],     # K: 100010
    'L': [1, 2, 3],  # L: 101010
    'M': [1, 3, 4],  # M: 110010
    'N': [1, 3, 4, 5], # N: 110110
    'O': [1, 3, 5],  # O: 100110
    'P': [1, 2, 3, 4], # P: 111010
    'Q': [1, 2, 3, 4, 5], # Q: 111110
    'R': [1, 2, 3, 5], # R: 101110
    'S': [2, 3, 4],  # S: 011010
    'T': [2, 3, 4, 5], # T: 011110
    'U': [1, 3, 6],  # U: 100011
    'V': [1, 2, 3, 6], # V: 101011
    'W': [2, 4, 5, 6], # W: 011101
    'X': [1, 3, 4, 6], # X: 110011
    'Y': [1, 3, 4, 5, 6], # Y: 110111
    'Z': [1, 3, 5, 6], # Z: 100111
    # Add more characters as needed
}

def run_braille_code(character):
    """Set GPIO pins according to the Braille pattern for the given character."""
    character = character.upper()
    if character in braille_mapping:
        pins_to_activate = braille_mapping[character]
        for pin in braille_pins.values():
            GPIO.output(pin, GPIO.LOW)  # Reset all pins
        for dot in pins_to_activate:
            GPIO.output(braille_pins[dot], GPIO.HIGH)  # Activate corresponding pins
        time.sleep(1)  # Keep the pin high for a second
        for pin in braille_pins.values():
            GPIO.output(pin, GPIO.LOW)  # Reset all pins after displaying the character
    else:
        print(f"Character '{character}' not found in Braille mapping")

def process_braille_file(file_path):
    """Process the file and run Braille code for each character."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            for char in line.strip():
                run_braille_code(char)

# Example usage
process_braille_file('braille_input.txt')

# Cleanup GPIO
GPIO.cleanup()

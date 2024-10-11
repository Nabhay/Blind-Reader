import RPi.GPIO as GPIO
import time
from smbus2 import SMBus
from Adafruit_ADS1x15 import ADS1115

def activate_braille_and_read(paragraph):
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)

    # Pin assignments for SMA actuators
    sma_pins = [17, 18, 27, 22, 23, 24]  # SMA pinouts for each bump
    # Set up the GPIO pins for SMA wires as output
    for pin in sma_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)  # Initialize all SMA to off

    # Initialize ADC (ADS1115) for thermosensor readings
    adc = ADS1115()
    GAIN = 1  # Gain setting for the ADC

    # Pin assignments for thermosensors (using ADC channels)
    thermo_pins = [0, 1, 2, 3, 4, 5]  # Analog pin assignments for temperature sensors

    # Motor control pins
    motor_x_step = 12  # Pin for X-axis step
    motor_x_dir = 16   # Pin for X-axis direction
    motor_y_step = 20  # Pin for Y-axis step
    motor_y_dir = 21   # Pin for Y-axis direction

    GPIO.setup(motor_x_step, GPIO.OUT)
    GPIO.setup(motor_x_dir, GPIO.OUT)
    GPIO.setup(motor_y_step, GPIO.OUT)
    GPIO.setup(motor_y_dir, GPIO.OUT)

    # Braille patterns for each letter (6-dot pattern for A-Z)
    braille_dict = {
        'A': [1, 0, 0, 0, 0, 0],
        'B': [1, 1, 0, 0, 0, 0],
        'C': [1, 0, 0, 1, 0, 0],
        'D': [1, 0, 0, 1, 1, 0],
        'E': [1, 0, 0, 0, 1, 0],
        'F': [1, 1, 0, 1, 0, 0],
        'G': [1, 1, 0, 1, 1, 0],
        'H': [1, 1, 0, 0, 1, 0],
        'I': [0, 1, 0, 1, 0, 0],
        'J': [0, 1, 0, 1, 1, 0],
        'K': [1, 0, 0, 0, 0, 1],
        'L': [1, 1, 0, 0, 0, 1],
        'M': [1, 0, 0, 1, 0, 1],
        'N': [1, 0, 0, 1, 1, 1],
        'O': [1, 0, 0, 0, 1, 1],
        'P': [1, 1, 0, 1, 0, 1],
        'Q': [1, 1, 0, 1, 1, 1],
        'R': [1, 1, 0, 0, 1, 1],
        'S': [0, 1, 0, 1, 0, 1],
        'T': [0, 1, 0, 1, 1, 1],
        'U': [1, 0, 0, 0, 0, 1],
        'V': [1, 1, 0, 0, 0, 1],
        'W': [0, 1, 0, 1, 1, 0],  # Special case for W
        'X': [1, 0, 0, 1, 0, 1],
        'Y': [1, 0, 0, 1, 1, 1],
        'Z': [1, 0, 0, 0, 1, 1]
    }

    def move_motors(x_steps, y_steps):
        """Move the motors to the desired position."""
        GPIO.output(motor_x_dir, GPIO.HIGH)  # Set motor direction for X
        for _ in range(x_steps):
            GPIO.output(motor_x_step, GPIO.HIGH)
            time.sleep(0.001)  # Pulse duration
            GPIO.output(motor_x_step, GPIO.LOW)
            time.sleep(0.001)

        GPIO.output(motor_y_dir, GPIO.HIGH)  # Set motor direction for Y
        for _ in range(y_steps):
            GPIO.output(motor_y_step, GPIO.HIGH)
            time.sleep(0.001)  # Pulse duration
            GPIO.output(motor_y_step, GPIO.LOW)
            time.sleep(0.001)

    try:
        # Continuously read temperature until it exceeds 25 degrees Celsius
        temperature = 0
        while temperature <= 25:
            thermosensor_values = []
            for i in thermo_pins:
                value = adc.read_adc(i, gain=GAIN)  # Read from the ADC
                thermosensor_values.append(value)

            # Convert ADC values to temperature (assuming a linear relationship for simplicity)
            temperature = (sum(thermosensor_values) / len(thermosensor_values)) * 0.1  # Example conversion
            print("Current temperature:", temperature)
            time.sleep(1)  # Wait before the next reading

        # Move to the desired position (example: move to (10, 10))
        move_motors(10, 10)

        # Activate braille for each letter in the paragraph
        for letter in paragraph:
            if letter in braille_dict:
                pattern = braille_dict[letter]
                for i in range(len(sma_pins)):
                    GPIO.output(sma_pins[i], pattern[i])
                time.sleep(1)  # Keep the braille active for 1 second
                # Turn off the SMA after activation
                for pin in sma_pins:
                    GPIO.output(pin, GPIO.LOW)
                time.sleep(0.5)  # Delay between letters

    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()  # Clean up GPIO on exit
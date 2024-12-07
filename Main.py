from pynput import keyboard
import threading
import subprocess
import time

# Variable to store valid keys
valid_keys = {"7", "8", "9", "4"}
# Variable to track if a script is running
script_running = False

# Function to handle keypress events
def on_press(key):
    global script_running
    try:
        if script_running:
            return  # Ignore keypresses while a script is running

        if key.char in valid_keys:  # Check if the key is valid
            print(f"Valid key pressed: {key.char}")
            script_running = True

            # Map keys to specific Python scripts and arguments
            script_mapping = {
                "7": ("Surroundings.py", []),
                "8": ("Read_Text.py", []),
                "9": ("Read_Text.py", ["-a"]),
                "4": ("Audio.py", []),
            }

            # Get the script and arguments for the pressed key
            script_to_run, script_args = script_mapping[key.char]
            print(f"Running script: {script_to_run} with arguments: {script_args}")

            # Run the corresponding script with arguments
            subprocess.run(["python", script_to_run] + script_args)  # Blocking call

            # Mark as not running after the script ends
            script_running = False
    except AttributeError:
        pass  # Ignore special keys like Shift, Ctrl, etc.

def on_release(key):
    global script_running
    if script_running:
        return False  # Stop listener

# Function to start listening for keypresses
def start_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# Start the listener in a background thread
listener_thread = threading.Thread(target=start_listener, daemon=True)
listener_thread.start()

# Main program doing other tasks
try:
    while True:
        time.sleep(1)  # Simulate work
except KeyboardInterrupt:
    print("\nExiting program.")

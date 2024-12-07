import pyaudio
import wave
import speech_recognition as sr
from pynput import keyboard
import os
import time
import google.generativeai as genai
from Secret_Parser import get_Secrets

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Create a directory for audio recordings
recordings_dir = "audio_recordings"
if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

# Generate timestamp and create the corresponding directory
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
recording_folder = os.path.join(recordings_dir, timestamp)
os.makedirs(recording_folder)

# Update the output filename to be inside the new folder
WAVE_OUTPUT_FILENAME = os.path.join(recording_folder, "output.wav")
RECORD_SECONDS = 10

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording...")

frames = []
def on_press(key):
    try:
        if key.char == 'q':  # Stop recording when 'q' is pressed
            print("Recording stopped.")
            return False
    except AttributeError:
        pass

# Collect audio data
with keyboard.Listener(on_press=on_press) as listener:
    while listener.running:
        data = stream.read(CHUNK)
        frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded data as a WAV file
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Transcribe the audio file
recognizer = sr.Recognizer()
with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
    audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print("Transcription: " + text)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"

# Save the transcription to a text file in the same folder
transcription_file = os.path.join(recording_folder, "transcription.txt")
with open(transcription_file, 'w') as file:
    file.write(text)

# Send the transcribed text to Gemini API
def send_to_gemini(text):
    secrets = get_Secrets()
    genai.configure(api_key=secrets["Google_Gemini_Api"])
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content(
        [text],
        generation_config=genai.types.GenerationConfig(temperature=1.0)
    )
    return response.text

# Save the Gemini response to a text file in the same folder
gemini_response = send_to_gemini(text)
gemini_file = os.path.join(recording_folder, "gemini_response.txt")
with open(gemini_file, 'w') as file:
    file.write(gemini_response)
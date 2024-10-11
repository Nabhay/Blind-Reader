---

# Blind Reader

**Blind Reader** is an AI-powered assistive tool designed for visually impaired users. It captures images, generates scene descriptions, and optionally converts these descriptions into Braille patterns using a hardware setup that involves **resistors**, **thermal sensors**, and **motors** to control a **Shape Memory Alloy (SMA)**. The SMA imprints Braille patterns on a 2D sheet, providing a tactile reading experience.

## Features

- **Image Capture**: The system captures real-time images via a camera.
- **AI-Powered Description**: It uses the Google Gemini AI API to generate detailed textual descriptions of the captured images.
- **Text-to-Speech (TTS)**: Automatically converts descriptions to speech via the `pyttsx3` library for audio output.
- **Braille Output**: Converts the description into Braille, printed on a 2D sheet using a hardware system driven by SMA, motors, and sensors.
- **2D Motorized Braille System**: A motor controls the position of the Braille display on a 2D plane, aligning the output for accurate reading.
- **GPU/CPU Mode**: Allows the user to choose whether image processing is performed on the CPU or GPU for performance optimization.

## Technologies Used

- **Python**: Core programming language for the project.
- **OpenCV**: Used for image capture and processing.
- **Google Generative AI (Gemini)**: Powers the image description functionality.
- **pyttsx3**: Text-to-Speech library that vocalizes the AI-generated descriptions.
- **RPi.GPIO and smbus2**: Control the GPIO pins on a Raspberry Pi, enabling hardware control for the thermal sensors, SMA, and motors.
- **Pillow**: Handles image manipulation and processing.
- **CRAFT**: Segments the image into smaller image segments.
- **EasyOCR**: Converts the segments into text

## Getting Started

### Hardware Requirements (Only if braille is activated)

1. **SMA actuators**: For generating tactile Braille patterns.
2. **Resistors**: For heating the SMA actuators.
3. **Thermal sensors**: To monitor and regulate the temperature of the SMA.
4. **Motors**: To control the movement of the Braille display across the 2D plane.
5. **Raspberry Pi**: To control the hardware via GPIO pins.

### Software Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nabhay/Blind-Reader.git
   ```
   
2. **Navigate to the Project Directory**:
   ```bash
   cd Blind-Reader
   ```

3. **Install Dependencies**:
   Install all necessary libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**:
   You’ll need a Google Gemini API key to enable image-to-text conversion. Add it to the `Secret.json` file:
   ```json
   {
     "Google_Gemini_Api": "YOUR_API_KEY"
   }
   ```

### Running the Project

- To capture an image, generate a description, and convert it into speech:
  ```bash
  python Description.py
  ```
  
- To enable Braille output:
  ```bash
  python Description.py --braille
  ```

- To use GPU for faster image processing:
  ```bash
  python Description.py --gpu
  ```

## File Descriptions

### main.py

`main.py` is the central hub of the Blind Reader system. It orchestrates image capture, AI-based description generation, TTS conversion, and optional Braille output. Key components include:

- **Image Capture**: Initiates the camera module to capture images.
- **CRAFT and EasyOCR**: Convert the image into text.
- **AI Description Generation**: Sends the image to the Google Gemini API, receiving a refined description in return.
- **Text-to-Speech**: Uses `pyttsx3` to vocalize the generated description.

### description.py

This script focuses on capturing images and generating textual descriptions using the Google Gemini API. It also handles converting the description into speech and optionally triggering the Braille system. Key processes:

- **Image Capture**: Utilizes OpenCV to capture images from the camera.
- **Google Gemini Integration**: Communicates with the API to receive a description of the captured image.
- **TTS Conversion**: The description is passed to `pyttsx3` for audio output.
- **Braille Output**: If Braille mode is enabled, it processes the description into Braille and controls the hardware to output the tactile pattern.

### braille.py

This file handles the physical Braille output. It converts text into Braille and uses hardware like **SMA actuators**, **resistors**, **thermal sensors**, and **motors** to create the tactile Braille pattern on a 2D surface. Key components:

- **Resistor Heating**: Activates resistors to heat the SMA, changing its shape to form Braille dots.
- **Temperature Monitoring**: Uses thermal sensors to ensure the SMA does not overheat during the process.
- **Motor Control**: Moves the Braille display across a 2D plane for accurate positioning.
- **Pattern Generation**: Converts the descriptive text into a corresponding Braille pattern and controls the hardware to imprint it on the sheet.

## Example Workflow

1. The system captures an image using a camera.
2. The system is converted to text by CRAFT and EasyOCR
3. The image is sent to the Google Gemini API, which generates a detailed description.
4. The description is converted into speech using `pyttsx3` for the user to hear.
5. If Braille mode is enabled, the description is translated into Braille and output onto the 2D surface using the SMA system.

## Requirements

`Python 3.8-3.9` is required to satisfy `requirements.txt`

Here’s the `requirements.txt` for the project:

```txt
opencv-python==4.5.1.48
numpy==1.24.3
torch==1.10.0+cu113
pillow==10.4.0
easyocr==1.7.1
google-generativeai==0.7.2
pyttsx3==2.91
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, whether they are related to software features or hardware optimizations.

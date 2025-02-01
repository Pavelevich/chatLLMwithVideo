# ChatGPT Vision - Camera Interaction Script
file:///home/pavel/PycharmProjects/camera/Pasted%20image.png

This script combines computer vision, face analysis, text-to-speech, and speech recognition to create an interactive experience. It uses a camera to detect faces and enables conversations powered by AI models like Whisper and GPT. Additionally, it includes features for emotion and facial characteristic analysis using DeepFace.

## Key Features

- **Face Detection**: Uses OpenCV with a Haar Cascade model to detect faces in real-time.
- **Face Analysis**: Leverages `DeepFace` to analyze facial characteristics (age, gender, emotion, and race).
- **Interactive Conversations**:
  - Text-to-speech conversion via `pyttsx3`.
  - Speech-to-text conversion using the `Whisper` model.
  - Responds to prompts or requests using a GPT model via a local endpoint.
- **YOLO Model**: A YOLOv8n instance is loaded (ready for object detection tasks if needed).
- **Audio Recording**: Records microphone audio and saves it as a WAV file.

## System Requirements

To use this script, you’ll need the following:

- **Hardware**:
  - A camera (e.g., Logitech or built-in camera).
  - Functional microphone.
  - GPU for optimal performance (some tasks require CUDA).

- **Software**:
  - Python 3.8+.
  - Required libraries (see the Installation section below).
  - Chat GPT API server running on `http://localhost:11434`.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <REPO_URL>
   cd <REPO_DIR>
   ```

2. **Install Dependencies**:
   Make sure to install the following Python libraries:
   - `opencv-python`
   - `opencv-python-headless`
   - `ultralytics`
   - `deepface`
   - `whisper`
   - `pyttsx3`
   - `requests`
   - `numpy`
   - `sounddevice`

   Use the following command to install them all at once:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Models**:
   - Download the YOLOv8n model (`yolov8n.pt`) from [Ultralytics](https://docs.ultralytics.com/).
   - Ensure the Haar Cascade model is available in your OpenCV installation.

4. **Additional Configurations**:
   - Verify that the GPT query service is running at the specified endpoint (`http://localhost:11434`).

## Usage

1. **Run the Script**:
   ```bash
   python all.py
   ```

2. **Controls**:
   - A video feed will appear from the camera, with detected faces highlighted by green rectangles.
   - Press `q` to exit the program.

3. **Interactive Flow**:
   - The system will greet you and ask you to speak.
   - Respond to the microphone; your speech will be transcribed into text and processed.
   - If you say "describe my face", the system will analyze your facial characteristics.
   - Otherwise, it will send your queries to a GPT model and respond accordingly.

## Key Dependencies

- **[Ultralytics YOLO](https://docs.ultralytics.com/)**: Used for object detection tasks.
- **[DeepFace](https://github.com/serengil/deepface)**: For emotion and facial characteristic analysis.
- **[OpenCV](https://opencv.org/)**: Essential for image and video capture and manipulation.
- **[Whisper](https://github.com/openai/whisper)**: Converts speech to text.
- **[Pyttsx3](https://pyttsx3.readthedocs.io/)**: Converts text to speech.

## Customization

- **Camera**: Modify the camera index in `cap = cv2.VideoCapture(2)` (e.g., change to `0` for a built-in camera).
- **Conversation Interval**: Adjust the waiting time between conversations by modifying the `conversation_interval` variable.
- **GPT Endpoint**: Update the GPT API URL in the function `chat_with_gpt()` if your endpoint is different.

## Notes

- If you encounter memory-related errors (OOM), try running certain models on the CPU instead of the GPU.
- Some libraries may require additional configurations based on your system.

## Credits
PAVEL CHMIRENKO

Developed using advanced AI and computer vision technologies such as OpenCV, YOLOv8, DeepFace, and GPT models.

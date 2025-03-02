#!/usr/bin/env  python3
import rospy
from std_msgs.msg import String
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import re

# EXPECTED_SAMPLES = 16000
# NOISE_FLOOR = 0.05  # Lowered the noise floor to make it more sensitive to quieter signals
# MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES / 8  # Reduced the required voice length
# Load the Whisper model (Choose from: "tiny", "base", "small", "medium", "large")
model = whisper.load_model("small")  # Adjust model size based on accuracy vs speed
audio_file_path = 'src/recorded_audio.wav'

def transcribe_audio_whisper(audio_file):
    """
    Uses Whisper to transcribe audio into text.
    """
    result = model.transcribe(audio_file)
    recognized_text = result["text"].lower()
    print(f"Recognized text: {recognized_text}")
    return recognized_text

def extract_keywords(text):
    """
    Extracts predefined keywords from transcribed text.
    """
    keywords_of_interest = ['left', 'right', 'forward', 'backward']
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    words = cleaned_text.split()
    extracted_keywords = [word for word in words if word in keywords_of_interest]
    return extracted_keywords

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()  # Wait until the recording is finished
    print("Recording complete!")
    return  audio

if __name__ == '__main__':

    # Example usage
    rospy.init_node("test_command_listener")
    rospy.loginfo("listener node has started.......")
    pub=rospy.Publisher("/command_topic",String,queue_size=10)
    while not rospy.is_shutdown():
        command=input()
        if(command=="r"):
            duration = 2  # seconds
            sample_rate = 16000  # Hz

            # Record audio for the given duration
            sd.stop()
            audio_data = record_audio(duration, sample_rate)

            # Save the recorded audio as a .wav file
            write(audio_file_path, sample_rate, audio_data)
            print(f"Audio saved as {audio_file_path}")
              # Replace with your actual file path
            recognized_text = transcribe_audio_whisper(audio_file_path)
            extracted_keywords = extract_keywords(recognized_text)
            print("Extracted keywords:", extracted_keywords)
            if len(extracted_keywords):
                print(extracted_keywords[0])
            else:
                print("No keywords detected")
#!/usr/bin/env  python3
import rospy
from std_msgs.msg import String
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import re
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops

# Define parameters
EXPECTED_SAMPLES = 16000
NOISE_FLOOR = 0.05  # Lowered the noise floor to make it more sensitive to quieter signals
MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES / 8  # Reduced the required voice length
BACKGROUND_NOISE_PATH = 'src/white_noise.wav'  # Example background noise path

# Path to the single .wav file
FILE_PATH = 'src/recorded_audio.wav'
# Load the full model directly from the saved file
model = tf.keras.models.load_model("src/best_model.keras")

# List of command words in categorical order
command_words = ['forward', 'backward', 'left', 'right', '_invalid']

# Function to preprocess the spectrogram and predict
def predict_spectrogram(model, spectrogram):
    # Resize spectrogram to match model's expected input shape (99, 43)
    spectrogram = tf.image.resize(spectrogram, (99, 43))  # Adjust to correct dimensions if needed
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension (99, 43, 1)
    spectrogram = np.expand_dims(spectrogram, axis=0)   # Add batch dimension (1, 99, 43, 1)

    # Make prediction using the model
    prediction = model.predict(spectrogram)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)

    # Map the index to the corresponding command word
    predicted_command = command_words[predicted_index]
    return predicted_command

# Function to get voice position
def get_voice_position(audio, noise_floor):
    audio = tf.cast(audio, tf.float32)  # Ensure audio is of type float32
    audio_mean = tf.reduce_mean(audio)  # Calculate mean (dtype will be promoted automatically)
    audio = audio - audio_mean  # Subtract the mean to center the audio
    audio = audio / tf.reduce_max(tf.abs(audio))  # Normalize to [-1, 1]
    print(f"Audio min: {tf.reduce_min(audio)}, Audio max: {tf.reduce_max(audio)}")  # Debugging line
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

# Work out how much of the audio file is actually voice
def get_voice_length(audio, noise_floor):
    position = get_voice_position(audio, noise_floor)
    return position[1] - position[0]

# Check if enough voice is present
def is_voice_present(audio, noise_floor, required_length):
    voice_length = get_voice_length(audio, noise_floor)
    print(f"Voice length: {voice_length}, Required length: {required_length}")  # Debugging line
    voice_length = tf.cast(voice_length, tf.float32)
    required_length = tf.cast(required_length, tf.float32)
    return voice_length >= required_length

# Process single file and generate spectrogram
def process_single_file(file_path, background_noise_path):
    # Load audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_path)

    # Get audio as numpy array and convert to float32
    audio = audio_tensor.to_tensor()  # Convert to Tensor

    # Normalize audio
    audio = tf.cast(audio, tf.float32)
    audio_mean = tf.reduce_mean(audio)  # Mean calculation without dtype argument
    audio = audio - audio_mean  # Subtract the mean
    audio = audio / tf.reduce_max(tf.abs(audio))  # Normalize to [-1, 1]

    # Check for voice presence
    if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
        print("No voice detected in audio file")
        return None

    # Visualize the audio waveform
    # plt.figure(figsize=(10, 4))
    # plt.plot(np.arange(len(audio)), audio.numpy())
    # plt.title("Audio Waveform")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Amplitude")
    # plt.show()

    # Random repositioning of audio
    voice_start, voice_end = get_voice_position(audio, NOISE_FLOOR)

    # Cast audio_shape to int64 for consistency with the voice positions
    audio_shape = tf.cast(tf.shape(audio)[0], tf.int64)
    voice_end = tf.cast(voice_end, tf.int64)  # Ensure voice_end is int64
    voice_start = tf.cast(voice_start, tf.int64)  # Ensure voice_start is int64

    end_gap = audio_shape - voice_end
    random_offset = np.random.uniform(0, voice_start + end_gap)
    audio = tf.roll(audio, shift=-int(random_offset) + int(end_gap), axis=0)

    # Add background noise
    background_tensor = tfio.audio.AudioIOTensor(background_noise_path)
    background = background_tensor.to_tensor()  # Convert to Tensor
    background_start = np.random.randint(0, tf.shape(background)[0] - EXPECTED_SAMPLES)
    background = background[background_start:background_start + EXPECTED_SAMPLES]
    background = tf.cast(background, tf.float32)
    background = background - tf.reduce_mean(background)
    background = background / tf.reduce_max(tf.abs(background))

    # Ensure background and audio have the same length (padding or trimming background)
    if tf.shape(audio)[0] != tf.shape(background)[0]:
        if tf.shape(audio)[0] > tf.shape(background)[0]:
            # Pad background noise if it's shorter
            padding = tf.zeros([tf.shape(audio)[0] - tf.shape(background)[0], 1], dtype=tf.float32)
            background = tf.concat([background, padding], axis=0)
        elif tf.shape(audio)[0] < tf.shape(background)[0]:
            # Trim background noise if it's longer
            background = background[:tf.shape(audio)[0]]

    # Add background noise with random volume scaling
    background_volume = np.random.uniform(0, 0.1)
    audio = audio + background_volume * background

    # Generate spectrogram
    spectrogram = audio_ops.audio_spectrogram(audio, window_size=320, stride=160, magnitude_squared=True).numpy()
    spectrogram = tf.nn.pool(
        input=tf.expand_dims(spectrogram, -1),
        window_shape=[1, 6],
        strides=[1, 6],
        pooling_type='AVG',
        padding='SAME'
    )
    spectrogram = tf.squeeze(spectrogram, axis=0)
    spectrogram = np.log10(spectrogram + 1e-6)

    return spectrogram



 # Function to record audio
def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete!")
    return np.squeeze(audio)

if __name__ == '__main__':
    rospy.init_node("command_sender")
    rospy.loginfo("Sender node has started.......")
    pub=rospy.Publisher("/command_topic",String,queue_size=10)
    # while not rospy.is_shutdown():
    #     command=input()
    #     if(command=="left"):
    #         pub.publish('left')
    
    rospy.loginfo("Press r to record")
    rospy.loginfo("Press q to quit")
    while not rospy.is_shutdown():
        command=input()
        if command=="q":
            break
        elif command=="r":
            # Parameters
            duration = 2  # seconds
            sample_rate = 16000  # Hz
             
            # Record audio for the given duration
            audio_data = record_audio(duration, sample_rate)
            
            # Save the recorded audio as a .wav file
            output_file = 'src/recorded_audio.wav'
            write(output_file, sample_rate, audio_data)
            print(f"Audio saved as {output_file}")
            # # Process and visualize the spectrogram for the single file
            # spectrogram = process_single_file(FILE_PATH, BACKGROUND_NOISE_PATH)
            # predicted_command = predict_spectrogram(model, spectrogram)
            # print(f"Predicted command: {predicted_command}")
            # pub.publish(predicted_command)
        else:
            rospy.logerr("Invalid command")
    rospy.logwarn("Test node ended")
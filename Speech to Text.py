# import speech_recognition as sr

# def live_speech_to_text():
#     recognizer = sr.Recognizer()
    
#     # Use the default microphone as the audio source
#     with sr.Microphone() as source:
#         print("Adjusting for ambient noise... Please wait.")
#         recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
#         print("You can start speaking now...")

#         while True:
#             try:
#                 # Capture audio from the microphone
#                 audio = recognizer.listen(source)
                
#                 # Recognize speech using Google Speech Recognition
#                 text = recognizer.recognize_google(audio)
                
#                 # Print the transcribed text
#                 print(f"You said: {text}")
            
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand the audio.")
#             except sr.RequestError as e:
#                 print(f"Error requesting results from Google Speech Recognition service; {e}")
#             except KeyboardInterrupt:
#                 print("Exiting...")
#                 break

# if __name__ == "__main__":
#     live_speech_to_text()





# import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import pyaudio
# import numpy as np

# # Load the Wav2Vec2 model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# # Initialize PyAudio to capture microphone input
# p = pyaudio.PyAudio()

# # Set microphone settings
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

# # Buffer to hold audio data
# audio_buffer = []

# def recognize_speech_from_microphone():
#     print("You can start speaking now...")

#     while True:
#         try:
#             # Capture audio from microphone
#             audio_data = stream.read(1024)
#             audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize the audio

#             # Append to the audio buffer
#             audio_buffer.extend(audio_array)

#             # Process every second (16000 samples)
#             if len(audio_buffer) >= 16000:  # Process every second of audio
#                 input_audio = np.array(audio_buffer[:16000])  # Take the first 16000 samples
#                 audio_buffer[:] = audio_buffer[16000:]  # Remove processed samples from buffer

#                 # Process the audio using the model
#                 inputs = processor(input_audio, return_tensors="pt", sampling_rate=16000)
#                 with torch.no_grad():
#                     logits = model(inputs.input_values).logits

#                 # Decode the predicted text
#                 predicted_ids = torch.argmax(logits, dim=-1)
#                 transcription = processor.decode(predicted_ids[0])

#                 # Print the transcription
#                 print(f"You said: {transcription}")

#         except Exception as e:
#             print(f"An error occurred: {e}")
#         except KeyboardInterrupt:
#             print("Exiting...")
#             break

# if __name__ == "__main__":
#     recognize_speech_from_microphone()







import speech_recognition as sr
import time

def live_speech_to_text():
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
        print("Ambient noise adjusted. You can start speaking now...")
        print("Say 'stop' to exit.")

        while True:
            try:
                # Capture audio from the microphone
                audio = recognizer.listen(source)
                
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio)
                
                # Print the transcribed text
                print(f"\nYou said: {text}")

                # Check if the user wants to stop the program
                if "stop" in text.lower():
                    print("Exiting...")
                    break
            
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Error requesting results from Google Speech Recognition service; {e}")
            except KeyboardInterrupt:
                print("Exiting...")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    live_speech_to_text()

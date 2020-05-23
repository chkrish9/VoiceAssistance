import speech_recognition as speech_recog
r = speech_recog.Recognizer()
with speech_recog.Microphone() as source:
    print("Say something")
    r.adjust_for_ambient_noise(source, duration=5)
    audio = r.listen(source)

try:
    text = r.recognize_sphinx(audio)
    print(text)
except:
    print("failed")

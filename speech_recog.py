import speech_recognition as speech_recog

r = speech_recog.Recognizer()
with speech_recog.Microphone() as source:
    print("Say something")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(text)
    except:
        print("failed")

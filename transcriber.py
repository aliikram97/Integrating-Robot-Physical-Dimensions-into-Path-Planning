import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
from playsound import playsound
def text_to_speech(text, lang='en'):
    result = GoogleTranslator(source='auto', target='en').translate(text)
    tts = gTTS(text=result, lang=lang)
    tts.save(r"C:\Users\Asus\Desktop\recorder/output.mp3")
    os.system(r"C:\Users\Asus\Desktop\recorder/start output.mp3")


def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing.....")
        # query = r.recognize_google(audio, language='urdu')
        query = r.recognize_google(audio,language='ur-PK')
        # query = r.recognize_sphinx(audio)
        print(f"The User said {query}\n")
    except Exception as e:
        print(f"say that again please.....{e}")
        return "None"
    return query


query = takecommand()
while (query == "None"):
    query = takecommand()
result = GoogleTranslator(source='auto', target='ps').translate(query)
text_to_speech(query)
path = r'C:\Users\Asus\Desktop\recorder/output.mp3'
playsound(path)





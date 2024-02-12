import speech_recognition as sr

# fetch audio from devices microphone
# and store in variable reference of type speech_recognition
a = sr.Recognizer()

# declaring device microphone as the source to take audio input
with sr.Microphone() as source:
	print("Say something!")

	# variable audio prints what user said in text format the end
	audio = a.listen(source)

# invoking sphinx for speech recognition
try:
	# printing audio
	print("You said " + a.recognize_sphinx(audio))

except sr.UnknownValueError:
	# if the voice is unclear
	print("Could not understand")

except sr.RequestError as e:
	print("Error; {0}".format(e))

from googletrans import Translator

translator = Translator()
result = translator.translate("Привет, как дела?", src="ru", dest="en")
print(result.text)

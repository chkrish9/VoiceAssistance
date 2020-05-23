from voice_model import VoiceModel
from voice_assistance import VoiceAssistance
while True:
    print("1. Retrain.")
    print("2. Chat.")
    print("3. Update Json.")
    print("4. Quit.")
    voice_ass = VoiceAssistance()
    inp = input("Choose option: ")
    if inp.lower() == "4":
        break
    if inp.lower() == "3":
        patters = []
        responses = []
        tag_name = input("Enter the tag name.")
        while True:
            pattern = input("Enter the question pattern. if your done with patterns type quit.")
            if pattern.lower() == "quit":
                break
            patters.append(pattern)
        while True:
            response = input("Enter the response pattern. if your done with patterns type quit.")
            if response.lower() == "quit":
                break
            responses.append(response)
        context_set = ""
        addJsonObj = {
          "tag": tag_name,
          "patterns": patters,
          "responses": responses,
          "context_set": context_set
        }
        voice_ass.update_json(addJsonObj)
    if inp.lower() == "1":
        VoiceModel.train_model()
    if inp.lower() == "2":
        voice_ass.load_model()
        voice_ass.load_data_set()
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break
            print(voice_ass.response(inp))

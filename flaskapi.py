from flask import Flask, jsonify, request
from voice_assistance import VoiceAssistance
from flask_cors import CORS
from voice_model import VoiceModel

app = Flask(__name__)
CORS(app)
voice_ass = VoiceAssistance()
voice_ass.load_model()
voice_ass.load_data_set()


@app.route('/', methods=['GET'])
def index():
    return jsonify(data="hi")


@app.route('/msg', methods=['POST'])
def get_resp():
    data = request.get_json()
    resp = voice_ass.response(data["msg"])
    return jsonify(data=resp)


@app.route('/retrain', methods=['GET'])
def retrain():
    VoiceModel.train_model()
    voice_ass.load_model()
    voice_ass.load_data_set()
    return jsonify("done")


@app.route('/update', methods=['POST'])
def update_json():
    data = request.get_json()
    voice_ass.update_json(data)
    return jsonify("updated")


if __name__ == "__main__":
    app.run(debug=False, threaded=False)

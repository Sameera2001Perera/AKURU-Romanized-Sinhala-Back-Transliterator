from flask import Flask, request, jsonify
from transliterator.transliteration import Transliterator
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time

app = Flask(__name__)

# Load transliteration model
model_directory = "models"
# model_directory = "Ransaka/sinhala-bert-medium-v2"
dictionary_path = "data/dictionary.txt"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForMaskedLM.from_pretrained(model_directory)
model.eval()
transliterator = Transliterator(
    dictionary_path=dictionary_path, tokenizer=tokenizer, model=model
)


@app.route("/", methods=["GET"])
def home():
    return open("templates/index.html").read()


@app.route("/transliterate", methods=["POST"])
def transliterate_text():
    data = request.get_json()  # Get JSON data from frontend
    singlish_text = data.get("text", "").strip()

    if not singlish_text:
        return jsonify({"error": "Input text is empty"}), 400

    # Perform transliteration
    time_now = time.time()
    print(f"Transliterating: {singlish_text}")
    sinhala_sentence = transliterator.generate_sinhala(singlish_text)
    time_taken = time.time() - time_now
    print(f"Time taken: {time_taken:.2f} seconds")

    # If multiple outputs exist, join them with a separator
    if isinstance(sinhala_sentence, list):
        sinhala_sentence = " | ".join(sinhala_sentence)

    return jsonify({"output": sinhala_sentence})  # Send JSON response


if __name__ == "__main__":
    app.run(debug=False, port=5000)

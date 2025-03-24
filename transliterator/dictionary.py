class TransliterationDictionary:
    def __init__(self, file_path):
        self.dictionary = self.load_dictionary(file_path)

    # Function to read the dictionary file and return a dictionary of Singlish to Sinhala words
    def load_dictionary(self, file_path):
        singlish_to_sinhala = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Split the line into Singlish word and Sinhala words part
                parts = line.split(", Sinhala Words: ")
                if len(parts) == 2:
                    singlish_word = parts[0].replace("Word: ", "").strip()
                    sinhala_words = eval(
                        parts[1].strip()
                    )  # Convert the string representation of list to an actual list
                    singlish_to_sinhala[singlish_word] = sinhala_words
        return singlish_to_sinhala

    def get(self, singlish_word):
        return self.dictionary.get(singlish_word, "Word not found")

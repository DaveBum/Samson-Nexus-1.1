import re
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import requests
import subprocess
import os
import threading
from tkinter import Tk, Button, Label, StringVar
from gtts import gTTS
from IPython.display import Audio, display

# Paths for the extracted EMO files
extract_dir = r'C:\Users\Davi\Downloads\EMO-main'

# Initialize models and tokenizers
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

emotion_model = DistilBertForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")
emotion_tokenizer = DistilBertTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")

memory = []

internet_access = False
stop_signal = False

def initialize_conversation():
    global internet_access
    response = input("Do you want to give Samson access to the internet? (yes/no): ").strip().lower()
    if response == 'yes':
        internet_access = True

def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        input_text = recognizer.recognize_google(audio)
        print(f"David: {input_text}")
        return input_text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, my speech service is down."

def detect_tone_and_sarcasm(input_text):
    tone = "neutral"
    sarcasm = False
    return tone, sarcasm

def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(input_ids=inputs, max_length=150, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = {result['word']: result['entity'] for result in ner_results}
    return entities

def validate_entities(entities):
    for entity in entities:
        if entities[entity] == 'O':  
            return False
    return True

def perform_web_search(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    search_results = response.json().get("Abstract", "No relevant information found.")
    return search_results

def analyze_sentence(input_text, sentence):
    input_embedding = similarity_model.encode(input_text, convert_to_tensor=True)
    sentence_embedding = similarity_model.encode(sentence, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(input_embedding, sentence_embedding).item()
    
    nlp_results = nlp_model(sentence, candidate_labels=["fact", "fabrication", "neutral"])
    nlp_label = nlp_results['labels'][0]

    entities = extract_entities(sentence)
    valid_entities = validate_entities(entities)

    if similarity < 0.5:
        return False

    if len(sentence.split()) < 3 or len(sentence.split()) > 50:
        return False

    if nlp_label == "fabrication":
        return False

    if not valid_entities:
        return False

    return True

def filter_response(input_text, response):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    valid_sentences = [sentence.strip() for sentence in sentences if analyze_sentence(input_text, sentence.strip())]
    return ' '.join(valid_sentences)

def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = emotion_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    emotion = torch.argmax(probabilities, dim=1).item()
    
    emotion_map = {
        0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "sadness", 5: "surprise", 6: "neutral",
        7: "love", 8: "trust", 9: "anticipation", 10: "anxiety", 11: "excitement", 12: "guilt",
        13: "shame", 14: "pride", 15: "envy", 16: "jealousy", 17: "contempt", 18: "embarrassment",
        19: "amusement", 20: "awe", 21: "contentment", 22: "relief", 23: "frustration", 24: "loneliness",
        25: "nostalgia", 26: "gratitude", 27: "compassion", 28: "sympathy", 29: "empathy",
        30: "curiosity", 31: "boredom", 32: "hope", 33: "confusion", 34: "regret", 35: "resentment",
        36: "suspicion", 37: "shock", 38: "astonishment", 39: "bitterness", 40: "desperation",
        41: "hostility", 42: "irritation", 43: "euphoria", 44: "melancholy", 45: "satisfaction",
        46: "serenity", 47: "agony", 48: "sorrow", 49: "bliss"
    }

    return emotion_map.get(emotion, "neutral")

def express_emotion(text, emotion):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    display(Audio("output.mp3"))
    run_talking_head(text, emotion)

def run_talking_head(text, emotion):
    emo_script_path = os.path.join(extract_dir, 'run_emo.py')
    subprocess.run(['python', emo_script_path, '--text', text, '--emotion', emotion])

def manage_conversation():
    global stop_signal
    initialize_conversation()

    while not stop_signal:
        input_text = convert_speech_to_text()
        memory.append(f"David: {input_text}")
        
        tone, sarcasm = detect_tone_and_sarcasm(input_text)

        attempts = 0
        max_attempts = 5
        valid_response = False
        response = None

        while attempts < max_attempts and not valid_response:
            generated_response = generate_response(input_text)
            filtered_response = filter_response(input_text, generated_response)

            if filtered_response:
                valid_response = True
                response = filtered_response
            else:
                attempts += 1

        if valid_response:
            emotion = detect_emotion(response)
            express_emotion(response, emotion)
            memory.append(f"Samson ({emotion}): {response}")

def start_conversation():
    global stop_signal
    stop_signal = False
    conversation_thread = threading.Thread(target=manage_conversation)
    conversation_thread.start()

def stop_conversation():
    global stop_signal
    stop_signal = True

# GUI setup
def setup_gui():
    root = Tk()
    root.title("Samson Conversational Agent")

    start_button = Button(root, text="Start Conversation", command=start_conversation)
    start_button.pack(pady=10)

    stop_button = Button(root, text="Stop Conversation", command=stop_conversation)
    stop_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()

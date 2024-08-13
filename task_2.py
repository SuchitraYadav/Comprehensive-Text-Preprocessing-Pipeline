import spacy
import contractions
import re
import emoji
import logging
import os
import nltk
from nltk.corpus import stopwords


logging.basicConfig(filename='text_preprocessingg.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

nltk.download('punkt')
# nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

# Load NLTK stop words
# nltk_stop_words = set(stopwords.words('english'))

class CustomStopWords:
    def __init__(self, additional_stopwords=None):
        self.stopwords = set(stopwords.words('english'))
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)

    def add_stopwords(self, words):
        self.stopwords.update(words)

    def remove_stopwords(self, words):
        self.stopwords.difference_update(words)

    def get_stopwords(self):
        return self.stopwords

def read_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        
        if not text:
            raise ValueError(f"File is empty: {file_path}")
        
        logging.info(f"Successfully read file: {file_path}")
        return text
    except UnicodeDecodeError as e:
        logging.error(f"Encoding error reading file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        raise

def write_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logging.info(f"Successfully wrote to file: {file_path}")
    except Exception as e:
        logging.error(f"Error writing to file {file_path}: {str(e)}")
        raise

def expand_contractions(text):
    try:
        expanded_text = contractions.fix(text)
        logging.debug(f"Expanded contractions: {expanded_text}")
        return expanded_text
    except Exception as e:
        logging.error(f"Error expanding contractions: {str(e)}")
        raise

def handle_negations(text):
        negations = {"not", "no", "never", "none", "neither", "nor", "cannot", "can't", "n't"}
        logging.info("The negated words to be identified:", negations)
        result_words = []
        doc = nlp(text)
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.lower_ in negations:
                result_words.append(token.text)
                i += 1
                if i < len(doc): 
                    next_token = doc[i]
                    # Append the next token with NOT
                    result_words.append("NOT_" + next_token.text)
                    i += 1
            else:
                result_words.append(token.text)
                i += 1
        return " ".join(result_words)

def handle_special_characters(text):
    try:
        text = emoji.replace_emoji(text, replace='')  # Remove emojis
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        logging.debug(f"Handled special characters: {text}")
        return text
    except Exception as e:
        logging.error(f"Error handling special characters: {str(e)}")
        raise

def tokenize_and_lemmatize(text):
    try:
        doc = nlp(text)
        lemmas = [token.text for token in doc]
        logging.debug(f"Tokenized and lemmatized: {lemmas}")
        return lemmas
    except Exception as e:
        logging.error(f"Error tokenizing and lemmatizing text: {str(e)}")
        raise

def preprocess_text(text, custom_stopwords):
    try:
        text = expand_contractions(text)
        text = handle_special_characters(text)
        text = handle_negations(text)
        words = tokenize_and_lemmatize(text)
        # words = [word for word in words if word.lower() not in custom_stopwords.get_stopwords()]
        preprocessed_text = ' '.join(words)
        logging.debug(f"Preprocessed text: {preprocessed_text}")
        return preprocessed_text
    except Exception as e:
        logging.error(f"Error in text preprocessing: {str(e)}")
        raise

def main(input_file, output_file, additional_stopwords=None):
    try:
        custom_stopwords = CustomStopWords(additional_stopwords)
        text = read_file(input_file)
        preprocessed_text = preprocess_text(text, custom_stopwords)
        write_file(output_file, preprocessed_text)
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    input_file = 'emojis.txt'  
    output_file = 'output.txt'  
    additional_stopwords = ['example', 'additional']  
    main(input_file, output_file, additional_stopwords)

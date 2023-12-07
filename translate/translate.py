import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from translate_benchmark import translate_benchmark

# https://github.com/facebookresearch/fairseq/tree/nllb     License: MIT
# https://huggingface.co/facebook/nllb-200-distilled-600M   License: CC-BY-NC ⚠️
models_dict = {
    'nllb-1.3B': 'facebook/nllb-200-1.3B',
    'nllb-3.3B': 'facebook/nllb-200-3.3B',
    'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M',
    'nllb-distilled-1.3B': 'facebook/nllb-200-distilled-1.3B',
}


def load_model(model_name):
    print('\tLoading model: %s' % model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(models_dict[model_name])
    tokenizer = AutoTokenizer.from_pretrained(models_dict[model_name])
    
    return (model, tokenizer)

#https://huggingface.co/spaces/Geonmo/nllb-translation-demo/blob/main/flores200_codes.py
source = 'eng_Latn'
target = 'zho_Hans'

model_name = 'nllb-distilled-600M'#'nllb-1.3B'#'nllb-3.3B'#'nllb-distilled-1.3B'

(model, tokenizer) = load_model(model_name)

translator = pipeline('translation', model=model,
                      tokenizer=tokenizer, src_lang=source, tgt_lang=target, device=0)
translator("warm up", max_length=16)[0]['translation_text']

def translate_print(text):
    trans = translator(text, max_length=len(text))[0]['translation_text']
    print(f"\033[93m{trans}\033[0m", end=' ')

# Regular expression to match 0, 1, or 2 leading multi-digit numbers followed by text
regex = r"^(\d+)?\s*(\d+)?\s*(.*)$"

# Define the end of sentence characters
end_of_sentence_chars = [".", "?", "!", ","]

# Function to find the first occurrence of any end of sentence character
def find_first_end_char_pos(text, end_of_sentence_chars):
    positions = [text.find(char) for char in end_of_sentence_chars if char in text]
    if positions:
        return min(positions)
    return -1

buffer = "" 

# Function to process each line of input
def process_line(line, end_of_sentence_chars):
    global buffer
    match = re.search(regex, line)
    if match:
        number1, number2, text = match.groups()
        #if number1:
        #    print(f"\033[93m{number1}\033[0m", end=' ')
        #if number2:
        #    print(f"\033[93m{number2}\033[0m", end=' ')
        print(f"{text}", end=' ')

        while text:
            first_end_char_pos = find_first_end_char_pos(text, end_of_sentence_chars)
            if first_end_char_pos != -1:
                # Include the end character in the sentence
                sentence, text = text[:first_end_char_pos + 1], text[first_end_char_pos + 1:]
                buffer += " " + sentence
                translate_print(buffer.strip())
                buffer = ""
            else:
                buffer += " " + text
                break                

        print()

import sys

for line in sys.stdin:#iter(sys.stdin.readline, b''):
    #print(f"[{line.strip()}]")
    process_line(re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line).strip(), end_of_sentence_chars)

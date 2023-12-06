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

model_name = 'nllb-1.3B'#'nllb-3.3B'#'nllb-distilled-600M'#'nllb-distilled-1.3B'

(model, tokenizer) = load_model(model_name)

translator = pipeline('translation', model=model,
                      tokenizer=tokenizer, src_lang=source, tgt_lang=target, device=0)
translator("warm up", max_length=16)[0]['translation_text']

translate_benchmark(
    lambda prompt: translator(prompt, max_length=len(prompt))[0]['translation_text'], 
    model_name
)
'''
s=[]
for prompt in source_text:
    tt = time.time()
    output = translator(prompt, max_length=len(prompt))
    s.append(time.time()-tt)
    output = output[0]['translation_text']
    print(f"{s[-1]:2.3f}s", output)
print(f"{model_name} mean={sum(s)/len(s):2.3f}")
'''
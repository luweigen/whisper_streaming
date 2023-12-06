from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianTokenizer, MarianMTModel
import torch
from translate_benchmark import translate_benchmark

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = 't5-small'
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_model.generate(
            t5_tokenizer.encode(f"warm up", return_tensors='pt').to(device)
        )

translate_benchmark(
    lambda prompt: t5_tokenizer.decode(
        t5_model.generate(
            t5_tokenizer.encode(f"translate English to Chinese:{prompt}", return_tensors='pt').to(device)
        )[0],
        skip_special_tokens=True
    ),
    model_name
)
'''
s=[]
for prompt in source_text:
    input_text = f"translate English to Chinese:{prompt}"
    tt = time.time()
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    outputs = t5_model.generate(input_ids)
    output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    s.append(time.time()-tt)
    print(f"{s[-1]:2.3f}s", output)
print(f"{model_name} mean={sum(s)/len(s):2.3f}")
'''

# https://huggingface.co/Helsinki-NLP/opus-mt-en-zh License: Apache-2.0 ✅
model_name = 'Helsinki-NLP/opus-mt-en-zh'

marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
marian_model = MarianMTModel.from_pretrained(model_name).to(device)
marian_model.generate(input_ids=marian_tokenizer("warm up", return_tensors='pt').to(device).input_ids)

translate_benchmark(
    lambda prompt: marian_tokenizer.decode(marian_model.generate(input_ids=marian_tokenizer(prompt, return_tensors='pt').to(device).input_ids)[0], skip_special_tokens=True), 
    model_name
)
'''
s=[]
for prompt in source_text:
    input_text = prompt
    tt = time.time()
    input_ids = marian_tokenizer(input_text, return_tensors='pt').input_ids
    outputs = marian_model.generate(input_ids=input_ids)
    output = marian_tokenizer.decode(outputs[0], skip_special_tokens=True)
    s.append(time.time()-tt)
    print(f"{s[-1]:2.3f}s", output)
print(f"{model_name} mean={sum(s)/len(s):2.3f}")
'''
#pip install transformers==4.33.1 torch sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from translate_benchmark import translate_benchmark

model_name = "THUDM/glm-large-chinese"#完全不支持中文"THUDM/glm-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
model = model.half().cuda()
model.eval()

# Inference
#inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
#inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
#inputs = inputs.to('cuda')
#outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
#print(tokenizer.decode(outputs[0].tolist()))

translate_benchmark(
    lambda prompt: tokenizer.decode(model.generate(**tokenizer.build_inputs_for_generation(tokenizer(
    f"English:'{prompt}'. 翻译成中文是:[MASK]", 
    return_tensors="pt"), max_gen_length=len(prompt)*2).to('cuda'), max_length=len(prompt)*2, eos_token_id=tokenizer.eop_token_id)[0].tolist()), 
    model_name
)

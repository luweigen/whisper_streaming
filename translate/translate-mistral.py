from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from translate_benchmark import translate_benchmark

#model_name = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
#model = model.to_bettertransformer()

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

sequences = pipe(
    "warm up",
    do_sample=True,
    max_new_tokens=10,
    temperature=0.1, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)

translate_benchmark(
    lambda prompt: pipe(
                        f"English:'{prompt}'. 翻译成中文:", 
                        do_sample=True, 
                        max_new_tokens=len(prompt), 
                        temperature=0.1, 
                        top_k=50, 
                        top_p=0.95, 
                        num_return_sequences=1
                    )[0]['generated_text'], 
    model_name
)
'''
s=[]
for prompt in source_text:
        prompt = f"English:'{prompt}'. 翻译成中文:"
        tt = time.time()
        sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=len(prompt),
            temperature=0.1, 
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
        )
        s.append(time.time()-tt)
        print(f"{s[-1]:2.3f}s", sequences[0]['generated_text'])
print(f"mean={sum(s)/len(s):2.3f}")
'''
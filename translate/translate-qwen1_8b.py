#pip install einops transformers_stream_generator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from translate_benchmark import translate_benchmark

# License: https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20RESEARCH%20LICENSE%20AGREEMENT Non-Commercial ⚠️
model_name = "Qwen/Qwen-1_8B"#"Qwen/Qwen-1_8B-Chat"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()

translate_benchmark(
    lambda prompt: tokenizer.decode(model.generate(**tokenizer(
        f"English:'{prompt}'. 翻译成中文是:", 
        return_tensors='pt').to(model.device)).cpu()[0], skip_special_tokens=True), 
    model_name
)
'''
s=[]
for prompt in source_text:
        prompt = f"English:'{prompt}'. 翻译成中文是:"
        tt = time.time()
        #response, history = model.chat(tokenizer, prompt, history=None)
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        s.append(time.time()-tt)
        print(f"{s[-1]:2.3f}s", response)
print(f"{model_name} mean={sum(s)/len(s):2.3f}")
'''
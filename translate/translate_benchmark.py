import time
from translate_source import source_text

def translate_benchmark(process_function, model_name):
    s = []
    for prompt in source_text:
        tt = time.time()
        output = process_function(prompt)
        s.append((time.time() - tt))
        print(f"{s[-1]:2.3f}s", output)
        #s.append((time.time() - tt)*1000/len(prompt))
        #print(f"{s[-1]:3.2f}ms/char", output)
    print(f"{model_name} mean={sum(s)/len(s):3.3f}")
#pip install mlx
#pip install huggingface_hub
#pip install tiktoken==0.3.3
from whisper_online import WhisperPipelineASR
import mlx.core as mx

import whisper
#from whisper import audio, decoding, load_models, transcribe

class MLXWhisperASR(WhisperPipelineASR):
    model_size_name = {
        "large-v3":"mlx-community/whisper-large-v3-mlx",
        "large-v3-4bit":"mlx-community/whisper-large-v3-mlx-4bit",
        "large-v2":"mlx-community/whisper-large-v2-mlx",
        "medium":"mlx-community/whisper-medium-mlx",
        "medium.en":"mlx-community/whisper-medium.en-mlx",
    }

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        self.path_or_hf_repo = self.model_size_name[modelsize]
        if self.original_language:
            self.transcribe_kargs["language"] = self.original_language
        return whisper

    def transcribe(self, audio, init_prompt=""):
        result = self.model.transcribe(
            audio,
            path_or_hf_repo=self.path_or_hf_repo,
            word_timestamps=True,
            **self.transcribe_kargs
        )        
        #print(result)
        '''
        {'text': ' So ⏩️ ourselves as ⏩️ faster.',
         'segments': [
             {'id': 0, 'seek': 0, 'start': 0.0, 'end': 5.68, 'text': ' So ⏩️ ourselves', 'tokens': [50365, ⏩️, 50665], 'temperature': 0.0, 'avg_logprob': -0.25580127239227296, 'compression_ratio': 1.3410852713178294, 'no_speech_prob': 0.1599762737751007, 'words': [
                 {'word': ' So', 'start': 0.0, 'end': 0.7, 'probability': 0.579}, 
                 ⏩️
                 {'word': ' ourselves', 'start': 5.16, 'end': 5.68, 'probability': 0.9956}]}, 
             {'id': 1, 'seek': 0, 'start': 5.68, 'end': 10.94, 'text': ' as ⏩️ faster.', 'tokens': [50665, ⏩️, 50915], 'temperature': 0.0, 'avg_logprob': -0.25580127239227296, 'compression_ratio': 1.3410852713178294, 'no_speech_prob': 0.1599762737751007, 'words': [
                 {'word': ' as', 'start': 5.68, 'end': 6.22, 'probability': 0.437}, 
                 ⏩️
                 {'word': ' faster.', 'start': 10.66, 'end': 10.94, 'probability': 0.999}]}],
          'language': 'en'
        }'''
        segments = []
        prev = 0.0
        def valid_sec(seconds):
            if not(isinstance(seconds, int) or isinstance(seconds, float)):
                seconds = prev
            else:
                prev = seconds
            return seconds
        for seg in result["segments"]:
            for word in seg["words"]:
                segments.append({"text":word["word"], "timestamp":(valid_sec(word["start"]),valid_sec(word["end"]))})
        return segments

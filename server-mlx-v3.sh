#!/bin/sh
python whisper_online_server.py --model large-v3 --backend mlx-whisper --language en --min-chunk-size 1 --port 43001 --samplerate 48000 --sampleencoding PCM_32 --buffer_trimming segment --buffer_trimming_sec 30

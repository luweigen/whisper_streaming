#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python whisper_online_server.py --model large-v3 --language en --min-chunk-size 1 --port 43001

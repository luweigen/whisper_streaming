#!/bin/sh
arecord -f S16_LE -r 16000 -c 1 --device="hw:0,0" -D default | nc localhost 43001 | python3 translate/translate.py

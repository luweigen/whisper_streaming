#!/bin/sh

#brew install sox
#sox -b 16 -L -r 16000 -c 1 -d -t wav - | nc localhost 43001
#can not change the 48000 sample rate and 32bit encoding
sox -L -c 1 -d -t wav - | nc localhost 43001

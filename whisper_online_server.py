#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
from colorama import Fore, Back, Style
import colorama
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)


# options from whisper_online
# TODO: code repetition

parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
parser.add_argument('--model', type=str, default='large-v2', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large,large-v3".split(","),help="Name size of the Whisper model to use (default: large-v3). The model is automatically downloaded from the model hub if not present in model cache dir.")
parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
parser.add_argument('--lan', '--language', type=str, default='en', help="Language code for transcription, e.g. en,de,cs.")
parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
parser.add_argument('--backend', type=str, default="faster-whisper", choices=["hf-pipeline","faster-whisper", "whisper_timestamped"],help='Load only this backend for Whisper processing.')
parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
args = parser.parse_args()


# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan

t = time.time()
print(f"Loading Whisper {size} model for {language}...",file=sys.stderr,end=" ",flush=True)

if args.backend == "faster-whisper":
    from faster_whisper import WhisperModel
    asr_cls = FasterWhisperASR
elif args.backend == "hf-pipeline":
    size = "openai/whisper-"+size
    asr_cls = WhisperPipelineASR
else:
    import whisper
    import whisper_timestamped
#    from whisper_timestamped_model import WhisperTimestampedASR
    asr_cls = WhisperTimestampedASR

asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)

if args.task == "translate":
    asr.set_translate_task()
    tgt_language = "en"
else:
    tgt_language = language

e = time.time()
print(f"done. It took {round(e-t,2)} seconds.",file=sys.stderr)

if args.vad:
    print("setting VAD filter",file=sys.stderr)
    asr.use_vad()


min_chunk = args.min_chunk_size
online = OnlineASRProcessor(asr,create_tokenizer(tgt_language))



demo_audio_path = "cs-maji-2.16k.wav"
if os.path.exists(demo_audio_path):
    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(demo_audio_path,0,1)

    # TODO: it should be tested whether it's meaningful
    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(a)
else:
    print("Whisper is not warmed up",file=sys.stderr)




######### Server objects

import line_packet
import socket

import logging


class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        r = self.conn.recv(self.PACKET_SIZE)
        return r


import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        while sum(len(x) for x in out) < self.min_chunk*SAMPLING_RATE:
            raw_bytes = self.connection.non_blocking_receive_audio()
            #print(raw_bytes[:10])
            #print(f"non_blocking_receive_audio() {len(raw_bytes)} bytes")#raw_bytes[:10])
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
            out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print(f"{Fore.GREEN}send_result:{Style.RESET_ALL}%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            #print(o,file=sys.stderr,flush=True)
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                print("break here",file=sys.stderr)
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                print("broken pipe -- connection closed?",file=sys.stderr)
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)




# Start logging.
level = logging.INFO
logging.basicConfig(level=level, format='whisper-server-%(levelname)s: %(message)s')

# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.host, args.port))
    s.listen(1)
    logging.info('INFO: Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logging.info('INFO: Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, online, min_chunk)
        proc.process()
        conn.close()
        logging.info('INFO: Connection to client closed')
logging.info('INFO: Connection closed, terminating.')

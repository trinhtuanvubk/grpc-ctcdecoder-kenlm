import os
import numpy as np 
from loguru import logger

from server.decoder import BeamCTCDecoder, GreedyDecoder
from server.phonetic_dict import vocab_lm_word




# Env for language model
LANGUAGE_MODEL_PATH = "./server/ngram_model/4gram_small.bin"
# print(type(os.getenv("BEAM_WIDTH")))
BEAM_WIDTH = 10
PAD_TOKEN = "<pad>"
CUTOFF_TOP_N = 40
CUTOFF_PROB = 1.0
NUM_PROCESSES = 8

# Define decoder
beam_decoder = BeamCTCDecoder(vocab_lm_word,
                              lm_path=LANGUAGE_MODEL_PATH,
                              cutoff_top_n=CUTOFF_TOP_N,
                              cutoff_prob=CUTOFF_PROB,
                              beam_width=BEAM_WIDTH,
                              num_processes=NUM_PROCESSES,
                              blank_index=vocab_lm_word.index(PAD_TOKEN)
                              )
greedy_decoder = GreedyDecoder(vocab_lm_word,
                               blank_index=vocab_lm_word.index(PAD_TOKEN)
                               )

def decode_logits(logits):
    # print("access here")
    logger.debug(type(logits))
    logger.debug("access here")
    # try: 
    logger.debug(logits.shape)
    greedy_decoded_output, greedy_decoded_offsets = greedy_decoder.decode(logits)
    greedy_decoded_offsets = greedy_decoded_offsets[0][0].tolist()
    greedy_trans = greedy_decoded_output[0][0]
    logger.debug("greedy trans: {}".format(greedy_trans))

    # beamsearch transcription
    beam_decoded_output, beam_decoded_offsets = beam_decoder.decode(logits)
    # beam_decoded_offsets = beam_decoded_offsets[0][0].tolist()
    beam_decoded_offsets = beam_decoded_offsets[0][0]
    logger.debug(beam_decoded_offsets)
    beam_trans = beam_decoded_output[0][0]
    logger.debug("beam trans: {}".format(beam_trans))

    return greedy_trans, beam_trans, beam_decoded_offsets

    # except: 
    #     bdo = np.ones((2,4))
    #     return "hihi", "haha", [bdo.tobytes()]
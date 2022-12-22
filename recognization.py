import numpy as np
from data_utils.data import DataGenerator
from adversarial_model import DeepSpeech2Model
import paddle.fluid as fluid
from scipy.signal import butter, lfilter
from adversarial_tools import *

data_generator = DataGenerator(vocab_filepath='./dataset/zh_vocab.txt',
                               mean_std_filepath='./dataset/mean_std.npz',
                               keep_transcription_text=False,
                               is_training=False)

adversarial_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                                num_conv_layers=2,
                                num_rnn_layers=3,
                                rnn_layer_size=2048,
                                use_gru=True,
                                share_rnn_weights=False,
                                place=fluid.CUDAPlace(0),
                                init_from_pretrained_model='models/step_final',
                                is_infer=True)


##########################
origin_path = 'result/4-6.wav'
infer_path = 'result/4-6.wav'
target_phrase = '朋友'
##########################

input_audio = load_wav(origin_path)
my_audio = load_wav(infer_path)
my_data, text = data_generator.process_utterance(infer_path, target_phrase)

probs, ctcloss = adversarial_model.infer(my_data, text)
print('ctcloss:',np.array(ctcloss)[0], '\nprobs shape:', probs.shape)

final_text = greedy_decode(probs_seq=probs,vocabulary=data_generator.vocab_list,blank_index=len(data_generator.vocab_list))
print('final_text decoded as: ', final_text)

corr = "{0:.4f}".format(np.corrcoef([input_audio, my_audio])[0][1])
print('Audio similarity to input: {}'.format(corr))
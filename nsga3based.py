import os
import numpy as np
from data_utils.data import DataGenerator
import random
from adversarial_model import DeepSpeech2Model
import paddle.fluid as fluid
from adversarial_tools import *
data_generator = DataGenerator(vocab_filepath='./dataset/zh_vocab.txt',
                               mean_std_filepath='./dataset/mean_std.npz',
                               keep_transcription_text=False,
                               is_training=False)
# paddle.enable_static()
adversarial_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                                num_conv_layers=2,
                                num_rnn_layers=3,
                                rnn_layer_size=2048,
                                use_gru=True,
                                share_rnn_weights=False,
                                place=fluid.CUDAPlace(0),
                                init_from_pretrained_model='models/step_final',
                                is_infer=True)

M = 3
pop_size = 100
best_text = ''
elite_size = 10
dist = float('inf')
prev_loss = None
mutation_p = 0.005
mu = 0.9
alpha = 0.001
elite_size = 10
noise_stdev = 0.02
##########################
max_iters = 400
dir_name = 'data'
audio_list = os.listdir(dir_name)
target_phrase = '不是'
##########################

for audio_path in audio_list:
    if audio_path[-4:] != '.wav':
        continue
    input_audio = load_wav(os.path.join(dir_name, audio_path)) # the result of first step
    Z,N = uniformpoint(pop_size,M)# Generate consistent reference solutions
    print('****************************current target_phrase:', target_phrase,'****************************')
    with open(os.path.join('result', 'mywords.txt'), 'a') as f:
        f.write(str(audio_path)+'\t'+target_phrase+'\n')
    best_text = ''
    prev_loss = None
    mutation_p = 0.005
    dist = float('inf')

    pop, pop_fun = create_init_pop(adversarial_model, target_phrase, os.path.join(dir_name, audio_path), pop_size)#生成初始种群及其适应度值
    Zmin = np.array(np.min(pop_fun,0)).reshape(1,M)# Calculate the ideal point


    itr = 1
    while itr <= max_iters and best_text != target_phrase:
        #计算得分
        pop_scores = pop_fun[:,0]
        elite_ind = np.argsort(pop_scores)[:elite_size]
        # elite_ind = np.argsort(pop_scores)[:]
        elite_pop, elite_fun = pop[elite_ind], pop_fun[elite_ind]
        elite_ctc = elite_fun[:,0]
        if prev_loss is not None and prev_loss != elite_ctc[0]:
            mutation_p = mu * mutation_p + alpha / np.abs(prev_loss - elite_ctc[0])

        if itr % 1 == 0:
            print('ITERATION {}'.format(itr))
            print('Current loss: {}'.format(elite_ctc[0]))

            best_pop = np.expand_dims(elite_pop[0],axis=0)
            ctc_scores, result_text = get_ctc_loss(adversarial_model, best_pop, target_phrase, data_generator, classify=True)
            best_text = result_text[0]
            dist = levenshteinDistance(best_text, target_phrase)
            corr = "{0:.4f}".format(np.corrcoef([input_audio, elite_pop[0]])[0][1])
            print('similarity: {}'.format(corr))
            print('Edit distance: {}'.format(dist))
            print('Currently decoded as: {}'.format(best_text))
            with open(os.path.join('result', audio_path[:-4]+'.txt'),'a') as f:
                f.write(str(elite_ctc[0])+'\t'+str(corr)+'\n')
            save_wav(best_pop[0], os.path.join('result',audio_path))
            if best_text == target_phrase: break

        next_pop = get_new_pop(elite_pop, elite_ctc, 200)
        off_pop = mutate_pop(next_pop, mutation_p, noise_stdev, elite_pop)
        prev_loss = elite_ctc[0]

        #mixpop
        off_fun = get_values(adversarial_model, os.path.join(dir_name, audio_path), off_pop, target_phrase) ###

        mix_pop, mix_fun = off_pop, off_fun
        Zmin = np.array(np.min(np.vstack((Zmin,off_fun)),0)).reshape(1,M)# Update the ideal point
        ###### NSGA3 Algorithm
        pop,Next = envselect(mix_pop,mix_fun,N,Z,Zmin,M)
        pop_fun = mix_fun[Next,:]
        
        itr += 1
    print('end one')
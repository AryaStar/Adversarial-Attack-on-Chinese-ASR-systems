import scipy.io.wavfile as wav
import os
from data_utils.speech import SpeechSegment
import numpy as np
from data_utils.data import DataGenerator
from data_utils.audio import AudioSegment
import soundfile
from model_utils.network import deep_speech_v2_network
from data_utils.augmentor.augmentation import AugmentationPipeline
import paddle
import paddle.inference as paddle_infer
from utils.predict import Predictor
from tqdm import tqdm
from adversarial_model import DeepSpeech2Model
import paddle.fluid as fluid
from data_utils.normalizer import FeatureNormalizer
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from itertools import groupby
from scipy.signal import butter, lfilter
from scipy.special import comb
from itertools import combinations
import copy
import math

def save_wav(samples, output_wav_file):
    audio =np.squeeze(samples)
    soundfile.write(output_wav_file, audio, 16000)
    print('Audio save to:', output_wav_file)
    

def load_wav(input_wav_file):
    samples = SpeechSegment.from_file(input_wav_file, '').samples
    rate = SpeechSegment.from_file(input_wav_file, '').sample_rate
    # print(speech_segment.samples.shape)
    return samples

def levenshteinDistance(s1, s2): 
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1) 
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def rms_db(samples):
    """返回以分贝为单位的音频均方根能量
    """
    # square root => multiply by 10 instead of 20 for dBs
    mean_square = np.mean(samples ** 2)
    return 10 * np.log10(mean_square)

#######################################################################################
def greedy_decode(probs_seq, vocabulary, blank_index=0):
    # 获得每个时间步的最佳索引
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    max_prob_list = [probs_seq[i][max_index_list[i]] for i in range(len(max_index_list)) if max_index_list[i] != blank_index]
    # 删除连续的重复索引和空索引，CTC的要求
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    index_list = [index for index in index_list if index != blank_index]
    # 索引列表转换为字符串
    text = ''.join([vocabulary[index] for index in index_list])
    return text

def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)

def specgram_real(samples, window_size, stride_size, sample_rate):
    """计算来自真实信号的频谱图样本"""
    # extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])
    # window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    return fft, freqs

def solve_input_samples(samples):
    '''
    将load_wav的矩阵结果处理为能被deepspeech2运行的输入矩阵
    即：对语音数据特征化和归一化
    '''
    gain = -20 - rms_db(samples)
    if gain > 300.0:
        raise ValueError(
            "无法将段规范化到 %f dB，因为可能的增益已经超过max_gain_db (%f dB)" % (-20, 300.0))
    gain = (min(300, -20 - rms_db(samples)))
    samples *= 10.**(gain / 20.)
    stride_ms=10.0
    window_ms=20.0
    max_freq=None
    sample_rate = 16000
    eps=1e-14
    """用 FFT energy计算线性谱图"""
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq不能大于采样率的一半")
    if stride_ms > window_ms:
        raise ValueError("stride_ms不能大于window_ms")
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    specgram, freqs = specgram_real(samples,
                window_size=window_size,
                stride_size=stride_size,
                sample_rate=sample_rate)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    features =  np.log(specgram[:ind, :] + eps)
    npzfile = np.load('dataset/mean_std.npz')
    mean = npzfile["mean"]
    std = npzfile["std"]
    result = (features - mean) / (std + eps)
    return result
def get_ctc_loss(model, pop, target_phrase, data_generator=None, classify=False):
    '''实现两个功能: 1）得到ctc_score就行 2)得到ctc_score和text_list'''
    ctc_score = []
    text_list = []
    tokens = list(target_phrase.strip())
    vocab_lines = []
    with open('dataset/zh_vocab.txt', 'r', encoding='utf-8') as file:
        vocab_lines.extend(file.readlines())
    vocab_list = [line[:-1] for line in vocab_lines]
    vocab_dict = dict(
        [(token, id) for (id, token) in enumerate(vocab_list)])
    target_phrase = [vocab_dict[token] for token in tokens]

    for p in tqdm(pop):
        my_data = solve_input_samples(p)
        probs, ctcloss = model.infer(my_data, target_phrase)
        ctc_score.append(np.array(ctcloss))
        if classify:
            final_text = greedy_decode(probs_seq=probs,vocabulary=data_generator.vocab_list,
                                    blank_index=len(data_generator.vocab_list))
    if classify:
        text_list.append(final_text)       
        return ctc_score, text_list
    
    return ctc_score

##############Genetic Algorithm##############
def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max())
    elite_pop_probs = scores_logits / scores_logits.sum()
    cand1 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5
    next_pop = mask * cand1 + (1-mask) * cand2
    return next_pop

def mutate_pop(pop, mutation_p, noise_stdev, elite_pop):# something need to think
    noise = np.random.randn(*pop.shape) * noise_stdev
    noise = highpass_filter(noise)
    mask = np.random.rand(pop.shape[0], elite_pop.shape[1]) < mutation_p
    new_pop = pop + noise * mask
    return new_pop

##############NSGA-3##############

def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))#按列合并
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N

def create_init_pop(adversarial_model,target_phrase,audio_path, pop_size):
    input_audio = load_wav(audio_path) #第一步结果
    pop = np.tile(input_audio, (pop_size, 1))
    # print('pop shape:', pop.shape)
    test_pop = np.expand_dims(pop[0],axis=0)
    ctc_score = get_ctc_loss(adversarial_model, test_pop, target_phrase)
    corr = np.corrcoef([input_audio, pop[0]])[0][1]
    popfun = np.tile(np.array((ctc_score[0][0],1-corr)),(pop_size,1))
    return pop, popfun

def get_values(model, audio_path, pop, target_phrase):
    input_audio = load_wav(audio_path)
    ctc_score = get_ctc_loss(model, pop, target_phrase)
    ctc_scores = np.array(ctc_score)
    corrs = []
    for i in pop:
        corr = 1-np.corrcoef([input_audio, i])[0][1]
        corrs.append(corr)
    corrs = np.array(corrs).reshape((len(corrs),1))
    popfun = np.hstack((ctc_scores, corrs))
    return popfun

def NDsort(mixpop,N,M):
    nsort = N#排序个数
    N,M = mixpop.shape[0],mixpop.shape[1]
    Loc1=np.lexsort(mixpop[:,::-1].T)#loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
    mixpop2=mixpop[Loc1]#排序后的矩阵情况
    Loc2=Loc1.argsort()#loc2为旧矩阵元素在新矩阵中的位置
    frontno=np.ones(N)*(np.inf)#初始化所有等级为np.inf
    # frontno[0]=1#第一个元素一定是非支配的
    maxfno=0#最高等级初始化为0
    while (np.sum(frontno < np.inf) < min(nsort,N)):#被赋予等级的个体数目不超过要排序的个体数目
        maxfno=maxfno+1
        for i in range(N):
            if (frontno[i] == np.inf):
                dominated = 0
                for j in range(i):
                    if (frontno[j] == maxfno):
                        m=0
                        flag=0
                        while (m<M and mixpop2[i,m]>=mixpop2[j,m]):
                            if(mixpop2[i,m]==mixpop2[j,m]):#相同的个体不构成支配关系
                                flag=flag+1
                            m=m+1 
                        if (m>=M and flag < M):
                            dominated = 1
                            break
                if dominated == 0:
                    frontno[i] = maxfno
    frontno=frontno#[Loc2]
    return frontno,maxfno

#求两个向量矩阵的余弦值,x的列数等于y的列数
def pdist(x,y):
    x0=x.shape[0]
    y0=y.shape[0]
    xmy=np.dot(x,y.T)#x乘以y
    xm=np.array(np.sqrt(np.sum(x**2,1))).reshape(x0,1)
    ym=np.array(np.sqrt(np.sum(y**2,1))).reshape(1,y0)
    xmmym=np.dot(xm,ym)
    cos = xmy/xmmym
    return cos

def lastselection(popfun1,popfun2,K,Z,Zmin):
    #选择最后一个front的解
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2)))-np.tile(Zmin,(popfun1.shape[0]+popfun2.shape[0],1))
    N,M = popfun.shape[0],popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]
    
    #正则化
    extreme = np.zeros(M)
    w = np.zeros((M,M))+1e-6+np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun/(np.tile(w[i,:],(N,1))),1))
    
    #计算截距
    extreme = extreme.astype(int)#python中数据类型转换一定要用astype
    #temp = np.mat(popfun[extreme,:]).I
    temp = np.linalg.pinv(np.mat(popfun[extreme,:]))
    hyprtplane = np.array(np.dot(temp,np.ones((M,1))))
    a = 1/hyprtplane
    if np.sum(a==math.nan) != 0:
        a = np.max(popfun,0)
    np.array(a).reshape(M,1)#一维数组转二维数组
    #a = a.T - Zmin
    a=a.T
    popfun = popfun/(np.tile(a,(N,1)))#进行normalize
    
    ##联系每一个解和对应向量
    #计算每一个解最近的参考线的距离
    cos = pdist(popfun,Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun**2,1))).reshape(N,1),(1,NZ))*np.sqrt(1-cos**2)
    #联系每一个解和对应的向量
    d = np.min(distance.T,0)
    pi = np.argmin(distance.T,0)
    
    #计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)
    
    #选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        #选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
#        I = np.ravel(np.array(np.where(choose == False)))
#        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (rho[j] == 0):
                s = np.argmin(d[N1+I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j]+1
        else:
            zchoose[j] = False
    return choose

def envselect(mix_pop,mixpop_fun,N,Z,Zmin,M):
    #非支配排序
    # Loc1,frontno,maxfno = NDsort(mixpop_fun,N,M)
    frontno,maxfno = NDsort(mixpop_fun,N,M)
    Next = frontno < maxfno
    # mix_pop = mix_pop[Loc1]
    # mixpop_fun = mixpop_fun[Loc1]
    #选择最后一个front的解
    Last = np.ravel(np.array(np.where(frontno == maxfno)))
    choose = lastselection(mixpop_fun[Next,:],mixpop_fun[Last,:],N-np.sum(Next),Z,Zmin)
    Next[Last[choose]] = True
    #生成下一代
    pop = copy.deepcopy(mix_pop[Next,:])
    return pop, Next


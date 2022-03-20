import os
import sys

import numpy as np
import paddle.inference as paddle_infer

from decoders.ctc_greedy_decoder import greedy_decoder


class Predictor:
    def __init__(self, model_dir, data_generator, decoding_method='ctc_greedy', alpha=1.2, beta=0.35,
                 lang_model_path=None, beam_size=10, cutoff_prob=1.0, cutoff_top_n=40, use_gpu=False, gpu_mem=500,
                 enable_mkldnn=False, num_threads=10):
        self.data_generator = data_generator
        self.decoding_method = decoding_method
        self.alpha = alpha
        self.beta = beta
        self.lang_model_path = lang_model_path
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        # 集束搜索方法的处理
        if decoding_method == "ctc_beam_search":
            try:
                from decoders.beam_search_decoder import BeamSearchDecoder
                self.beam_search_decoder = BeamSearchDecoder(alpha, beta, lang_model_path,
                                                             data_generator.vocab_list)
            except ModuleNotFoundError:
                raise Exception('缺少ctc_decoders库，如果是Windows系统，请使用ctc_greedy。如果是Linux系统，且一定要使用ctc_beam_search解码策略'
                                '请执行`cd decoders && sh setup.sh`编译ctc_beam_search解码函数')

        # 创建 config
        self.config = paddle_infer.Config(os.path.join(model_dir, 'model.pdmodel'),
                                          os.path.join(model_dir, 'model.pdiparams'))
        self.config.enable_use_gpu(1000, 0)
        self.config.enable_memory_optim()

        if use_gpu:
            self.config.enable_use_gpu(gpu_mem, 0)
        else:
            self.config.disable_gpu()
            self.config.set_cpu_math_library_num_threads(num_threads)
            if enable_mkldnn:
                self.config.set_mkldnn_cache_capacity(10)
                self.config.enable_mkldnn()

        # enable memory optim
        self.config.enable_memory_optim()
        self.config.disable_glog_info()

        # 根据 config 创建 predictor
        self.predictor = paddle_infer.create_predictor(self.config)

        # 获取输入层
        self.audio_data_handle = self.predictor.get_input_handle('audio_data')
        self.seq_len_data_handle = self.predictor.get_input_handle('seq_len_data')
        self.masks_handle = self.predictor.get_input_handle('masks')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()

        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path)
        else:
            print('预热文件不存在，忽略预热!', file=sys.stderr)

    # 预测
    def predict(self, audio_path):
        # 加载音频文件，并进行预处理
        audio_feature = self.data_generator.process_utterance(audio_path, "")
        audio_len = audio_feature[0].shape[1]
        mask_shape0 = (audio_feature[0].shape[0] - 1) // 2 + 1
        mask_shape1 = (audio_feature[0].shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                                    (1, mask_shape0, mask_max_len)), 32, axis=0)

        audio_data = np.array(audio_feature[0]).astype('float32')[np.newaxis, :]
        seq_len_data = np.array([audio_len]).astype('int64')
        masks = np.array(mask).astype('float32')[np.newaxis, :]

        # 设置输入
        self.audio_data_handle.reshape([audio_data.shape[0], audio_data.shape[1], audio_data.shape[2]])
        self.seq_len_data_handle.reshape([audio_data.shape[0]])
        self.masks_handle.reshape([masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]])
        self.audio_data_handle.copy_from_cpu(audio_data)
        self.seq_len_data_handle.copy_from_cpu(seq_len_data)
        self.masks_handle.copy_from_cpu(masks)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        self.output_data = output_handle.copy_to_cpu()

        # 执行解码
        if self.decoding_method == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search(probs_split=self.output_data,
                                                                 beam_alpha=self.alpha,
                                                                 beam_beta=self.beta,
                                                                 beam_size=self.beam_size,
                                                                 cutoff_prob=self.cutoff_prob,
                                                                 cutoff_top_n=self.cutoff_top_n,
                                                                 vocab_list=self.data_generator.vocab_list,
                                                                 blank_id=len(self.data_generator.vocab_list))
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=self.output_data,
                                    vocabulary=self.data_generator.vocab_list,
                                    blank_index=len(self.data_generator.vocab_list))

        score, text = result[0], result[1]
        return score, text

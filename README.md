# Adversarial-Attack-on-Chinese-ASR-systems

* To generate adversarial examples for your own files, please ensure that the file is sampled at 16KHz and uses signed 16-bit ints as the data type. Our method is based on multi-objective evolutionary algorithm with three evaluated objectives, namely, CTC loss, speech similarity, and speech signal-to-noise ratio.
* Datasets of our experiments: We selected the datasets THCHS-30 and AISHELL-1. We randomly select 100 audio samples in wav format from each of these two datasets as the experimental subjects of our adversarial attack.
* Chinese ASR System: The Chinese ASR system we selected is DeepSpeech2 developed by Baidu.

# Attack on Chinese ASR system: DeepSpeech2(PaddleSpeech)

1. Ensure to Install DeepSpeech2 system first.
   One of the Implementations for DeepSpeech2 can be find [here](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech). This project is developed based on the DeepSpeech2 project based on PaddlePaddle. The paper of DeepSpeech2 is [&#34;Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin&#34;](http://proceedings.mlr.press/v48/amodei16.pdf). The project supports for training and prediction under Windows, Linux, and support for development board reasoning predictions such as NVIDIA Jetson.
2. Our adversarial attack code are detailed in adversarial_tools.py and nsga3based.py. Please copy files: sadversarial_tools.py, nsga3based.py and adversarial_model.py into the DeepSpeech2 project directory.

   ```
   python nsga3based.py
   ```

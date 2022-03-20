# Adversarial-Attack-on-Chinese-ASR-systems

* Datasets: We selected the datasets THCHS-30 and AISHELL-1. We randomly select 100 audio samples in wav format from each of these two datasets as the experimental subjects of our adversarial attack.

* Chinese ASR Systems: The Chinese ASR system we selected is DeepSpeech2 developed by Baidu. its support for Chinese Mandarin is excellent 

* Our method is based on multi-objective evolutionary algorithm with three evaluated objectives, namely, CTC loss, speech similarity, and speech signal-to-noise ratio.

  Our adversarial attack code is detailed in adversarial_tools.py and nsga3based.py


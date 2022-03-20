# Adversarial-Attack-on-Chinese-ASR-systems

(1) Datasets: We selected the datasets THCHS-30 and AISHELL-1. We randomly select 100 audio samples in wav format from each of these two datasets as the experimental subjects of our adversarial attack.

(2) Chinese ASR Systems: The Chinese ASR system we selected is DeepSpeech2 developed by Baidu [8]. This system predicts Chinese characters directly based on an end-to-end character-level language model, and its support for Chinese Mandarin is excellent [8]. The only information we can access to is the probability of each frame of its output. We do not need access to the internal structure and weights of the model.


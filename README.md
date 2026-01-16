# Adversarial-Attack-on-Chinese-ASR-systems
* ✨ View and run the project on [AI studio](https://aistudio.baidu.com/projectdetail/3502090)!
* To generate adversarial examples for your own files, please ensure that the file is sampled at 16KHz and uses signed 16-bit ints as the data type. Our method is based on multi-objective evolutionary algorithm with three evaluated objectives, namely, CTC loss, speech similarity, and speech signal-to-noise ratio.
* Datasets of our experiments: We selected the datasets [THCHS-30](https://arxiv.org/abs/1512.01882) and [AISHELL-1](https://ieeexplore.ieee.org/abstract/document/8384449/). We randomly select 100 audio samples in wav format from each of these two datasets as the experimental subjects of our adversarial attack.
* Chinese ASR System: The Chinese ASR system we selected is DeepSpeech2 developed by Baidu.

# Attack on Chinese ASR System: DeepSpeech2(PaddleSpeech)

* Ensure to Install DeepSpeech2 system first. One of the Implementations for DeepSpeech2 can be find [here](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech). This project is developed based on the DeepSpeech2 project based on PaddlePaddle. The paper of DeepSpeech2 is [&#34;Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin&#34;](http://proceedings.mlr.press/v48/amodei16.pdf). The project supports for training and prediction under Windows, Linux, and support for development board reasoning predictions such as NVIDIA Jetson.
* Please copy files: sadversarial_tools.py, nsga3based.py and adversarial_model.py into the DeepSpeech2 project directory. Now create and run an attack:

  ```
  python nsga3based.py
  ```
* We use this script to recognized the audio by DeepSpeech2 in roder to verify that the attack succeeded:

  ```
  python recognization.py
  ```

# Chinese Adversarial Samples

We encourage readers to listen to our chinese audio adversarial examples and the original one in the [attacking_samples](attacking_samples) directory.

## Revise Chinese Words in a Sentence

The `chinese_audio.wav` will be recognized as `"想听歌曲父亲"`  and `adversarial_audio.wav` will be recognized as `"想听歌曲母亲"` by the DeepSpeech2 system.

```
成功加载了预训练模型：models/step_final
ctcloss: 0.94573337 
final_text decoded as:  想听歌曲母亲
Audio similarity to input: 0.9966
```

## Adversarial Attack on Chinese Phrase

The `chinese_audio_phrase.wav` will be recognized as `"可怜好哦"`  and `adversarial_audio_phrase.wav` will be recognized as `"取款机"` by the DeepSpeech2 system.

```
成功加载了预训练模型：models/step_final
ctcloss: 11.0739765
final_text decoded as: 取款机
Audio similarity to input: 0.8445
```

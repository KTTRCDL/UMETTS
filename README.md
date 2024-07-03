# MM-TTS: A Unified Framework for Multimodal, Prompt-Induced Emotional Text-to-Speech Synthesis

* Coming soon: Our team is actively working on the latest code updates to provide better performance and functionality. Stay tuned as the new version will be released soon!

### Xiang Li, Zhi-Qi Cheng, Jun-Yan He, Xiaojiang Peng, Alexander G. Hauptmann

In our recent [paper](https://arxiv.org/abs/2404.18398), we propose MM-TTS: A Unified Framework for Multimodal, Prompt-Induced Emotional Text-to-Speech Synthesis.

Emotional Text-to-Speech (E-TTS) synthesis has gained significant attention in recent years due to its potential to enhance human-computer interaction. However, current E-TTS approaches often struggle to capture the complexity of human emotions, primarily relying on oversimplified emotional labels or single-modality inputs. To address these limitations, we propose the Multimodal Emotional Text-to-Speech System (MM-TTS), a unified framework that leverages emotional cues from multiple modalities to generate highly expressive and emotionally resonant speech. MM-TTS consists of two key components: (1) the Emotion Prompt Alignment Module (EP-Align), which employs contrastive learning to align emotional features across text, audio, and visual modalities, ensuring a coherent fusion of multimodal information; and (2) the Emotion Embedding-Induced TTS (EMI-TTS), which integrates the aligned emotional embeddings with state-of-the-art TTS models to synthesize speech that accurately reflects the intended emotions. Extensive evaluations across diverse datasets demonstrate the superior performance of MM-TTS compared to traditional E-TTS models. Objective metrics, including Word Error Rate (WER) and Character Error Rate (CER), show significant improvements on ESD dataset, with MM-TTS achieving scores of 7.35% and 3.07%, respectively. Subjective assessments further validate that MM-TTS generates speech with emotional fidelity and naturalness comparable to human speech. 

<!-- Visit our [demo]() for audio samples and we also provide the [pretrained models](). -->

<img src="assets/framework.png">

## Pre-requisites

1. NVIDIA GPU
2. Python >= 3.7

## Setup
1. Clone this repository
    ```shell
    # SSH
    git clone --recursive git@github.com:KTTRCDL/MMTTS.git

    # HTTPS
    git clone --recursive https://github.com/KTTRCDL/MMTTS.git
    ```
2. Install python requirements. Please refer [requirements.txt](requirements.txt) for the complete list of dependencies.
    ```shell
    # requirements.txt
    pip install -r requirements.txt
    # CLIP
    pip install EPAlign/CLIP
    ```
3. Download datasets
    - Download and extract the Emotion Speech Dataset (ESD) following the instructions in the official repository [Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data)
    - Download and extract the Real-world Expression Database (RAF-DB) following the instructions in the official website [Real-world Affective Faces Database](http://www.whdeng.cn/raf/model1.html)
    - Download and extract the [Multimodal EmotionLines Dataset (MELD)](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
4. Preprocess the datasets
    - ESD: follow the jupyter notebook [preprocess/ESD.ipynb](preprocess/ESD.ipynb)
    <!-- - MELD: follow the jupyter notebook [preprocess/MELD.ipynb](preprocess/MELD.ipynb) -->
## Train & Finetune
### Emotion Prompt Alignment Module (EP-Align)
1. Train the model
    
    follow the jupyter notebook [EPAlign/script/EPAlign_prompt_audio_finetune.ipynb](EPAlign/script/EPAlign_prompt_audio_finetune.ipynb), [EPAlign/script/EPAlign_prompt_vision_finetune.ipynb](EPAlign/script/EPAlign_prompt_vision_finetune.ipynb) and [EPAlign/script/EPAlign_finetune.ipynb](EPAlign/script/EPAlign_finetune.ipynb)

2. Extract the aligned emotional features
    
    follow the jupyter notebook [EPAlign/script/extract_emofeature.ipynb](EPAlign/script/extract_emofeature.ipynb)

### Emotion Embedding-Induced TTS (EMI-TTS)
1. Train the model
    ```shell
    # Variant VITS
    # Cython-version Monotonoic Alignment Search
    cd EMITTS/VITS/model/monotonic_align
    python setup.py build_ext --inplace
    cd ../..
    # path/to/json e.g. config/esd_en_e5.json, 
    python train.py -c path/to/json -m esd_en

    # Variant FastSpeech2
    cd EMITTS/FastSpeech2
    # need to change some path config in EMITTS/FastSpeech2/config/config.py file
    python -m src.scripts.train
    ```

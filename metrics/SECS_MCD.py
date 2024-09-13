import torch
from tqdm import tqdm
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from mel_cepstral_distance import get_metrics_wavs
from resemblyzer import VoiceEncoder, preprocess_wav

def load_filelist(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filelist = [line.strip().split(split) for line in f]
    return filelist

def resample_wav(wav_path, sr, new_path):
    wav, _ = torchaudio.load(wav_path)
    wav = T.Resample(orig_freq=sr, new_freq=16000)(wav)
    torchaudio.save(new_path, wav, 16000)

if __name__ == '__main__':
    # Configurations
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    audiopaths_sid_text_efeature_path = "/path/to/audiopaths_sid_text_efeature.txt"
    GT_audio_folder = "/path/to/GT_audio_folder"
    gen_audio_folder = "/path/to/gen_audio_folder"

    # load data
    audiopaths_sid_text_efeature = load_filelist(audiopaths_sid_text_efeature_path)

    # load VoiceEncoder model
    encoder = VoiceEncoder()

    # calculate Speaker Embedding Cosine Similarity
    similaritys = []
    for i in tqdm(range(len(audiopaths_sid_text_efeature))):
        file_path = audiopaths_sid_text_efeature[i][0]
        GT_path = GT_audio_folder + file_path.split("/")[-1]
        gen_path = gen_audio_folder + file_path.split("/")[-1]
        wav_GT = preprocess_wav(GT_path)
        wav_gen = preprocess_wav(gen_path)
        embed_GT = encoder.embed_utterance(wav_GT)
        embed_gen = encoder.embed_utterance(wav_gen)
        similarity = embed_GT @ embed_gen
        similaritys.append(similarity)
    similaritys = np.array(similaritys)
    print("median speaker embedding cosine similarity: ", np.median(similaritys))

    # calculate mel cepstral distance
    mcds = []
    for i in tqdm(range(len(audiopaths_sid_text_efeature))):
        file_path = audiopaths_sid_text_efeature[i][0]
        GT_path = GT_audio_folder + file_path.split("/")[-1]
        gen_path = gen_audio_folder + file_path.split("/")[-1]
        gen_new_path = gen_audio_folder + file_path.split("/")[-1].replace("test", "test_16k")
        resample_wav(gen_path, 16000, gen_new_path)
        mcd = get_metrics_wavs(Path(GT_path), Path(gen_new_path))
        mcds.append(mcd)
    print("median mcd: ", np.median(mcds))
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import jiwer
import whisper
import pandas as pd
from whisper.normalizers import EnglishTextNormalizer

def load_filelist(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filelist = [line.strip().split(split) for line in f]
    return filelist

def load_wav(path, sample_rate=16000):
    waveform, sample_rate_origion = torchaudio.load(path)
    if sample_rate_origion != sample_rate:
        # change sample rate
        waveform = torchaudio.transforms.Resample(sample_rate_origion, sample_rate)(waveform)
    return waveform, sample_rate

def remove_empty_strings(lst):
    return [item for item in lst if item.strip()]
def change_empty_strings_to_none(lst):
    return [item if item.strip() else "1" for item in lst]

class SPEECH_DATASET(Dataset):
    def __init__(self, audiopaths_text="/path/to/audiopaths_sid_text_efeature.txt",
                 device="cuda", wavrootdir="/path/to/wavrootdir"):
        self.audiopaths_text = load_filelist(audiopaths_text)
        self.device = device
        self.wavrootdir = wavrootdir

    def __len__(self):
        return len(self.audiopaths_text)
    
    def __getitem__(self, index):
        audiopath = self.wavrootdir + self.audiopaths_text[index][0].split('/')[-1]
        text = self.audiopaths_text[index][2]
        audio, sample_rate = load_wav(audiopath, sample_rate=16000)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)
    
if __name__ == '__main__':
    # Configurations
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    audiopaths_sid_text_efeature_path = "/path/to/audiopaths_sid_text_efeature.txt"
    gen_audio_folder = "/path/to/gen_audio_folder"
    normalizer = EnglishTextNormalizer()

    # load data
    dataset = SPEECH_DATASET(audiopaths_text=audiopaths_sid_text_efeature_path, device=DEVICE, wavrootdir=gen_audio_folder)
    loader = DataLoader(dataset, batch_size=16)

    # load whisper model
    model = whisper.load_model("base.en")
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    # predict without timestamps for short-form transcription
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    
    # whisper predict
    hypotheses = []
    references = []

    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    # calculate WER and CER
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    wer = jiwer.wer(change_empty_strings_to_none(list(data["reference_clean"])), change_empty_strings_to_none(list(data["hypothesis_clean"])))
    cer = jiwer.cer(change_empty_strings_to_none(list(data["reference_clean"])), change_empty_strings_to_none(list(data["hypothesis_clean"])))
    print(f"WER: {wer * 100:.7f} %")
    print(f"CER: {cer * 100:.7f} %")

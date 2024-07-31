import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

import librosa


class TextMelLoader(torch.utils.data.Dataset):
    """ modify
        1) loads audio,text pairs (add speaker id, emotion label and emotion feature here)
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text_emotion, hparams, is_return_path=False):
        self.audiopaths_sid_text_emotion = load_filepaths_and_text(audiopaths_sid_text_emotion)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        # modify
        self.is_multi_speaker = hparams.is_multi_speaker
        self.is_multi_emotion = hparams.is_multi_emotion
        self.is_emotion_feature = hparams.is_emotion_feature
        self.is_return_path = is_return_path

        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_sid_text_emotion)

    def get_mel_text_pair(self, audiopath_and_text):
        
        if self.is_multi_speaker and self.is_multi_emotion and self.is_emotion_feature:
            audiopath, sid, text, emotion = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
            sid = self.get_speaker_id(sid)
            eid = self.get_emotion_id(emotion)
            efeature = self.get_emotion_feature(emotion)
            if self.is_return_path:
                return (text, mel, sid, eid, efeature, audiopath)
            return (text, mel, sid, eid, efeature)
        elif self.is_multi_speaker and self.is_multi_emotion:
            audiopath, sid, text, emotion = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
            sid = self.get_speaker_id(sid)
            eid = self.get_emotion_id(emotion)
            return (text, mel, sid, eid)
        elif self.is_multi_speaker:
            audiopath, sid, text= audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
            sid = self.get_speaker_id(sid)
            return (text, mel, sid)
        else:
            # separate filename and text
            audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
            return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                # raise ValueError("{} {} SR doesn't match target {} SR".format(
                #     sampling_rate, self.stft.sampling_rate))
                # modify
                audio = librosa.resample(
                audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate)
                audio = torch.FloatTensor(audio.astype(np.float32))
                sampling_rate = self.sampling_rate
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_speaker_id(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid
    
    def get_emotion_id(self, eid):
        if type(eid) == str and 'pt' in eid:
            efeature2eid = {
                'neutral': 0,
                'angry': 1,
                'happy': 2,
                'sad': 3,
                'surprise': 4,
            }
            eid = efeature2eid[eid.split('/')[-1][:-3]]
        else:
            eid = torch.IntTensor([int(eid)])
        return eid

    def get_emotion_feature(self, eid):
        # feature = torch.load(featurepath)
        if type(eid) == str and 'pt' in eid:
            feature = torch.load(eid,  map_location=torch.device('cpu'))
        else:
            eid2efeature_name={
                0: 'neutral',
                1: 'angry',
                2: 'happy',
                3: 'sad',
                4: 'surprise',
            }
            featurepath = f'path/to/mmefeatures/{eid2efeature_name[eid]}.pt'
            feature = torch.load(featurepath,  map_location=torch.device('cpu'))
        return feature

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_sid_text_emotion[index])

    def __len__(self):
        return len(self.audiopaths_sid_text_emotion)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # modify
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, sid, eid, efeature]
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # modify
        if len(batch[0]) == 3:
            sid = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                sid[i] = batch[ids_sorted_decreasing[i]][2]
            return text_padded, input_lengths, mel_padded, gate_padded, \
                output_lengths, sid
        elif len(batch[0]) == 4:
            sid = torch.LongTensor(len(batch))
            eid = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                sid[i] = batch[ids_sorted_decreasing[i]][2]
                eid[i] = batch[ids_sorted_decreasing[i]][3]
            return text_padded, input_lengths, mel_padded, gate_padded, \
                output_lengths, sid, eid
        elif len(batch[0]) == 5:
            sid = torch.LongTensor(len(batch))
            eid = torch.LongTensor(len(batch))
            efeature = torch.FloatTensor(len(batch), 512)
            for i in range(len(ids_sorted_decreasing)):
                sid[i] = batch[ids_sorted_decreasing[i]][2]
                eid[i] = batch[ids_sorted_decreasing[i]][3]
                efeature[i, :] = batch[ids_sorted_decreasing[i]][4]
            return text_padded, input_lengths, mel_padded, gate_padded, \
                output_lengths, sid, eid, efeature
        else:
            return text_padded, input_lengths, mel_padded, gate_padded, \
                output_lengths

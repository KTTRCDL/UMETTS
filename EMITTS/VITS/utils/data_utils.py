import random
import os
import torch
import librosa

import numpy as np

from torch.utils.data import Dataset, distributed

import utils.commons as commons
import utils.text_utils as text_utils
import utils.utils as utils
import utils.mel_utils as mel_utils


class TextAudioSpeakerEmotionfeatureLoader(Dataset):
    """
        1) loads text, audio(spectrogram), audio(wav), speaker_id, emotion_feature
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text_emotion, hparams, seed=42) -> None:
        super().__init__()
        self.audiopaths_sid_text_emotion = utils.load_filelist(
            audiopaths_sid_text_emotion)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        if hasattr(hparams, 'is_emotion_feature'):
            self.is_emotion_feature = hparams.is_emotion_feature 
        else:
            self.is_emotion_feature = True

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(seed)
        random.shuffle(self.audiopaths_sid_text_emotion)
        self._filter()

    def __getitem__(self, index):
        return self.get_audio_text_speaker_emotion_pair(self.audiopaths_sid_text_emotion[index])

    def __len__(self):
        return len(self.audiopaths_sid_text_emotion)

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_efeaturepaths_new = []
        lengths = []
        for audiopath, sid, text, efeaturepath in self.audiopaths_sid_text_emotion:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_efeaturepaths_new.append(
                    [audiopath, sid, text, efeaturepath])
                lengths.append(os.path.getsize(audiopath) //
                               (2 * self.hop_length))
        self.audiopaths_sid_text_emotion = audiopaths_sid_text_efeaturepaths_new
        self.lengths = lengths

    def get_audio_text_speaker_emotion_pair(self, audiopath_sid_text_emotion):
        # separate audio filename, speaker_id, text, emotion, emotion feature path
        audiopath = audiopath_sid_text_emotion[0]
        sid = audiopath_sid_text_emotion[1]
        text = audiopath_sid_text_emotion[2]
        emotion = audiopath_sid_text_emotion[3]

        spec, wav = self.get_audio(audiopath)
        sid = self.get_speaker_id(sid)
        text = self.get_text(text)
        if self.is_emotion_feature:
            efeature = self.get_emotion_feature(emotion)
            return (text, spec, wav, sid, efeature)
        else:
            eid = self.get_emotion_id(emotion)
            return (text, spec, wav, sid, eid)

    def get_audio(self, filename):
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            audio = librosa.resample(
                audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate)
            audio = torch.FloatTensor(audio.astype(np.float32))
            sampling_rate = self.sampling_rate
            # raise ValueError("{} SR doesn't match target {} SR".format(
            #     sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = mel_utils.spectrogram_torch(audio_norm, self.filter_length,
                                               self.sampling_rate, self.hop_length, self.win_length,
                                               center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_speaker_id(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = text_utils.cleaned_text_to_sequence(text)
        else:
            text_norm = text_utils.text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_emotion_id(self, eid):
        eid = torch.LongTensor([int(eid)])
        return eid

    def get_emotion_feature(self, featurepath):
        # feature = torch.load(featurepath)
        feature = torch.load(featurepath,  map_location=torch.device('cpu'))
        return feature


class DistributedBucketSampler(distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                # self.boundaries.pop(i+1)
                self.boundaries.pop(i)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket %
                   total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(
                    len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * \
                (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j *
                                                           self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class TextAudioSpeakerEmotionfeatureCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False, efeature_dim=256, is_emotion_feature=True):
        self.return_ids = return_ids
        self.efeature_dim = efeature_dim
        self.is_emotion_feature = is_emotion_feature
    
    def __call__(self, batch):
        """Collate's training batch from normalized text, audio, speaker identities, emotion features
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, efeature]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        if self.is_emotion_feature:
            emofeature = torch.FloatTensor(len(batch), self.efeature_dim)
        else:
            eid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            if self.is_emotion_feature:
                emofeature[i, :] = row[4]
            else:
                eid[i] = row[4]
        if self.is_emotion_feature:
            if self.return_ids:
                return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, emofeature, ids_sorted_decreasing
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, emofeature
        else:
            if self.return_ids:
                return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid, ids_sorted_decreasing
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid

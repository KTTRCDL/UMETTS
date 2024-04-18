import tensorflow as tf
from text import symbols

class HParamsDict:
    def __init__(self, hparams_dict):
        self.hparams_dict = hparams_dict

    def __getattr__(self, name):
        if name in self.hparams_dict:
            return self.hparams_dict[name]()
        else:
            raise AttributeError(f"'HParamsDict' object has no attribute '{name}'")

def hparams_to_dict(*args, **kwargs):
    hparams_dict = {}
    for hparams in args:
        for name in dir(hparams):
            if not name.startswith("__"):
                value = getattr(hparams, name)
                if not callable(value):
                    hparams_dict[name] = value
    for name, value in kwargs.items():
        if not callable(value):
            hparams_dict[name] = value
    return hparams_dict


def create_hparams(hparams_string=None, 
                   verbose=False, 
                   is_multi_speaker=False,
                   is_multi_emotion=False,
                   is_emotion_feature=False,
                   training_files='filelists/ljs_audio_text_train_filelist.txt',
                   validation_files='filelists/ljs_audio_text_val_filelist.txt',):
    """Create model hyperparameters. Parse nondefault from given string."""
    # 'epochs': lambda: 500,

    hparams = HParamsDict({
    'epochs': lambda: 2500,
    'iters_per_checkpoint': lambda: 1000,
    'seed': lambda: 42,
    'dynamic_loss_scaling': lambda: True,
    'fp16_run': lambda: False,
    'distributed_run': lambda: False,
    'dist_backend': lambda: "nccl",
    'dist_url': lambda: "tcp://localhost:54321",
    'cudnn_enabled': lambda: True,
    'cudnn_benchmark': lambda: False,
    'ignore_layers': lambda: ['embedding.weight'],
    'load_mel_from_disk': lambda: False,
    'training_files': lambda: training_files,
    'validation_files': lambda: validation_files,
    'text_cleaners': lambda: ['english_cleaners'],
    'max_wav_value': lambda: 32768.0,
    'sampling_rate': lambda: 22050,
    'filter_length': lambda: 1024,
    'hop_length': lambda: 256,
    'win_length': lambda: 1024,
    'n_mel_channels': lambda: 80,
    'mel_fmin': lambda: 0.0,
    'mel_fmax': lambda: 8000.0,
    'n_symbols': lambda: len(symbols),
    'symbols_embedding_dim': lambda: 512,
    'encoder_kernel_size': lambda: 5,
    'encoder_n_convolutions': lambda: 3,
    'encoder_embedding_dim': lambda: 512,
    'n_frames_per_step': lambda: 1,
    'decoder_rnn_dim': lambda: 1024,
    'prenet_dim': lambda: 256,
    'max_decoder_steps': lambda: 1000,
    'gate_threshold': lambda: 0.5,
    'p_attention_dropout': lambda: 0.1,
    'p_decoder_dropout': lambda: 0.1,
    'attention_rnn_dim': lambda: 1024,
    'attention_dim': lambda: 128,
    'attention_location_n_filters': lambda: 32,
    'attention_location_kernel_size': lambda: 31,
    'postnet_embedding_dim': lambda: 512,
    'postnet_kernel_size': lambda: 5,
    'postnet_n_convolutions': lambda: 5,
    'use_saved_learning_rate': lambda: False,
    'learning_rate': lambda: 1e-3,
    'weight_decay': lambda: 1e-6,
    'grad_clip_thresh': lambda: 1.0,
    'batch_size': lambda: 64,
    'mask_padding': lambda: True,
    'is_multi_speaker': lambda: is_multi_speaker,
    'is_multi_emotion': lambda: is_multi_emotion,
    'is_emotion_feature': lambda: is_emotion_feature,
    'n_speakers': lambda: 11,
    'speaker_embedding_dim': lambda: 512,
    'n_emotions': lambda: 5,
    'emotion_feature_dim': lambda: 512,
    'emotion_embedding_dim': lambda: 512,
    })

    # hparams = tf.contrib.training.HParams(
    #     ################################
    #     # Experiment Parameters        #
    #     ################################
    #     epochs=500,
    #     iters_per_checkpoint=1000,
    #     seed=1234,
    #     dynamic_loss_scaling=True,
    #     fp16_run=False,
    #     distributed_run=False,
    #     dist_backend="nccl",
    #     dist_url="tcp://localhost:54321",
    #     cudnn_enabled=True,
    #     cudnn_benchmark=False,
    #     ignore_layers=['embedding.weight'],

    #     ################################
    #     # Data Parameters             #
    #     ################################
    #     load_mel_from_disk=False,
    #     training_files='filelists/ljs_audio_text_train_filelist.txt',
    #     validation_files='filelists/ljs_audio_text_val_filelist.txt',
    #     text_cleaners=['english_cleaners'],

    #     ################################
    #     # Audio Parameters             #
    #     ################################
    #     max_wav_value=32768.0,
    #     sampling_rate=22050,
    #     filter_length=1024,
    #     hop_length=256,
    #     win_length=1024,
    #     n_mel_channels=80,
    #     mel_fmin=0.0,
    #     mel_fmax=8000.0,

    #     ################################
    #     # Model Parameters             #
    #     ################################
    #     n_symbols=len(symbols),
    #     symbols_embedding_dim=512,

    #     # Encoder parameters
    #     encoder_kernel_size=5,
    #     encoder_n_convolutions=3,
    #     encoder_embedding_dim=512,

    #     # Decoder parameters
    #     n_frames_per_step=1,  # currently only 1 is supported
    #     decoder_rnn_dim=1024,
    #     prenet_dim=256,
    #     max_decoder_steps=1000,
    #     gate_threshold=0.5,
    #     p_attention_dropout=0.1,
    #     p_decoder_dropout=0.1,

    #     # Attention parameters
    #     attention_rnn_dim=1024,
    #     attention_dim=128,

    #     # Location Layer parameters
    #     attention_location_n_filters=32,
    #     attention_location_kernel_size=31,

    #     # Mel-post processing network parameters
    #     postnet_embedding_dim=512,
    #     postnet_kernel_size=5,
    #     postnet_n_convolutions=5,

    #     ################################
    #     # Optimization Hyperparameters #
    #     ################################
    #     use_saved_learning_rate=False,
    #     learning_rate=1e-3,
    #     weight_decay=1e-6,
    #     grad_clip_thresh=1.0,
    #     batch_size=64,
    #     mask_padding=True  # set model's padded outputs to padded values
    # )

    if hparams_string:
        # tf.logging.info('Parsing command line hparams: %s', hparams_string)
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        # tf.logging.info('Final parsed hparams: %s', hparams.values())
        print('Final parsed hparams: %s', hparams.values())

    return hparams

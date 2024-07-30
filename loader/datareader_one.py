import numpy as np
import soundfile as sf
import librosa
import torch as th

def audio(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data, sr, fs)
    return wave_data

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def parse_scp(scp, path_list):
    with open(scp) as fid: 
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({"inputs": tmp[0], "duration": tmp[1]})
            else:
                path_list.append({"inputs": tmp[0]})

class DataReader(object):
    def __init__(self, filename, noisy_id, clean_id, aux_segment, sample_rate): # filename是不带id的待解码音频，noisy_id是带id的带解码音频，clean是带id的注册音频
        self.file_list = []
        self.utt2spk = dict()
        self.spk2aux = dict()
        parse_scp(filename, self.file_list)
        self.get_utt2spk(noisy_id)
        self.get_spk2utt(clean_id)
        self.aux_segment_length = aux_segment * sample_rate

    def extract_feature(self, path):
        path = path["inputs"]
        spk_id = self.utt2spk[path]
        aux_path = self.spk2aux[spk_id]
        utt_id = path.split("/")[-1]
        data = get_firstchannel_read(path).astype(np.float32)
        aux1_data = get_firstchannel_read(aux_path).astype(np.float32)

        max_norm = np.max(np.abs(data))
        max_aux_norm = np.max(np.abs(aux1_data))
        if max_norm == 0:
            max_norm = 1
        if max_aux_norm == 0:
            max_aux_norm = 1
        data = data / max_norm
        aux1_data = aux1_data / max_aux_norm

        inputs = np.reshape(data, [1, data.shape[0]])
        aux1_inputs = np.reshape(aux1_data, [1, aux1_data.shape[0]])

        inputs = th.from_numpy(inputs)
        aux1_inputs = th.from_numpy(aux1_inputs)
        egs = {
            "mix": inputs,
            "utt_id": utt_id,
            "aux": aux1_inputs,
            "max_norm": max_norm
        }
        return egs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

    def get_utt2spk(self, path):
        lines = open(path, "r").readlines()
        for line in lines:
            line = line.strip().split()
            utt_path, spk_id = line[0], line[1]
            self.utt2spk[utt_path] = spk_id
    
    def get_spk2utt(self, path):
        lines = open(path, "r").readlines()
        for line in lines:
            line = line.strip().split()
            utt_path, spk_id = line[0], line[1]
            self.spk2aux[spk_id] = utt_path
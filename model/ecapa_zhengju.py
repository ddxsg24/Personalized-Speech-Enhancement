import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import typing
# from packaging import version


class TdnnConvLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 512,
                 context_size: int = 5,
                 dilation: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dropout_p: float =0.0,
                 groups: int = 1,
                 activation=nn.ReLU):
        super(TdnnConvLayer, self).__init__()

        self.activation = activation()
        self.kernel = nn.Conv1d(input_dim, output_dim,
                                kernel_size=context_size,
                                dilation=dilation,
                                stride=stride,
                                padding=padding,
                                groups=groups)
        self.norm = nn.BatchNorm1d(output_dim)

        self.dropout = None
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.norm(self.activation(self.kernel(x)))

        if self.dropout is not None:
            x = self.dropout(x)

        return x


def compute_statistics(x, m, dim: int =2, eps: float =1e-12):
    mean = (m * x).sum(dim)
    std = torch.sqrt(
        (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
    )
    return mean, std


def length_to_mask(length, max_len: int, dtype: typing.Optional[torch.dtype]=None, device: typing.Optional[torch.device]=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()

    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    
    return mask


class AttentiveStatisticsPooling(nn.Module):
    expansion = 2

    def __init__(self, in_channel, attentive_channel, global_context=True):
        super(AttentiveStatisticsPooling, self).__init__()

        self.eps = 1e-12
        self.global_context = global_context

        if global_context:
            self.tdnn = TdnnConvLayer(in_channel * 3, # for concat([])
                                      attentive_channel,
                                      1, 1)
        else:
            self.tdnn = TdnnConvLayer(in_channel,
                                      attentive_channel,
                                      1, 1)

        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attentive_channel, in_channel, kernel_size=1)

    def forward(self, x, lengths: typing.Optional[torch.Tensor]=None):
        L = x.shape[-1]

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = compute_statistics(x, mask / total, eps=self.eps)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = compute_statistics(x, attn, eps=self.eps)

        pooled_stats = torch.cat([mean, std], dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)
        return pooled_stats


class Res2DilatedConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=8,
                 kernel_size=3,
                 dilation=1):
        super(Res2DilatedConv1d, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channels = in_channels // scale
        hidden_channels = out_channels // scale
        
        self.conv_list = nn.ModuleList(
            [
                TdnnConvLayer(in_channels, hidden_channels, kernel_size, dilation=dilation, padding=int(dilation * (kernel_size - 1) / 2))
                for _ in range(scale - 1)
            ]
        )
        self.conv_list.insert(0, nn.Identity())
        self.scale = scale

    def forward(self, x):
        # print('len of conv_list', len(self.conv_list))
        # print('x', x.shape)
        y = []
        y_i = torch.zeros(1)
        temp = torch.chunk(x, self.scale, dim=1)
        for index, layers in enumerate(self.conv_list):
            x_i = temp[index]
            if index == 0:
                y_i = x_i
            elif index == 1:
                y_i = layers(x_i)
            else:
                y_i = layers(x_i + y_i)
            # print('y_{}'.format(index+1), y_i.shape)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        # print('out', y.shape)
        return y


class SEBlock1d(nn.Module):
    """
        SEBlock for the 3-D data
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, se_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs, lengths: typing.Optional[torch.Tensor]=None):
        """
            inputs: batch_size feat_dim frame_dim
        """
        batch_size, feat_dim, frame_num = inputs.shape

        if lengths is not None:
            mask = length_to_mask(lengths * frame_num, max_len=frame_num, device=inputs.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            outputs = (inputs * mask).sum(dim=2, keepdim=True) / total
        else:
            outputs = inputs.mean(dim=2, keepdim=True)

        outputs = self.fc(outputs)
        return outputs * inputs


class SERes2Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=8,
                 se_channels=128,
                 kernel_size=3,
                 dilation=1,
                 activation=nn.ReLU):
        super(SERes2Block, self).__init__()

        self.block1 = TdnnConvLayer(
            input_dim=in_channels,
            output_dim=out_channels,
            context_size=1,
            dilation=1,
            activation=activation
        )
        self.dilated_conv = Res2DilatedConv1d(
            out_channels, out_channels, scale, kernel_size, dilation
        )
        self.block2 = TdnnConvLayer(
            input_dim=out_channels,
            output_dim=out_channels,
            context_size=1,
            dilation=1,
            activation=activation
        )
        self.se_block = SEBlock1d(
            out_channels, se_channels, out_channels
        )

        self.short_cut = None
        if in_channels != out_channels:
            self.short_cut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1
            )

    def forward(self, x, lengths: typing.Optional[torch.Tensor]=None):
        residual = x
        if self.short_cut is not None:
            residual = self.short_cut(x)

        x = self.block1(x)
        x = self.dilated_conv(x)
        x = self.block2(x)

        x = self.se_block(x, lengths)

        return x + residual


class STFT(nn.Module):
    def __init__(self,
                 window_size: int = 400,
                 padded_window_size: int = 512,
                 window_shift: int = 160,
                 window_type: str = "povey",
                 snip_edges: bool = True,
                 raw_energy: bool = True,
                 blackman_coeff: float = 0.42,
                 energy_floor: float = 1.0,
                 dither: float = 0.0,
                 remove_dc_offset: bool = True,
                 preem_coeff: float = 22.0
                 ):
        super().__init__()
        self.window_size = window_size
        self.padded_window_size = padded_window_size
        self.window_shift = window_shift
        self.window_type = window_type
        self.snip_edges = snip_edges
        self.raw_energy = raw_energy
        self.blackman_coeff = blackman_coeff
        self.energy_floor = energy_floor
        self.dither = dither
        self.remove_dc_offset = remove_dc_offset
        self.preem_coeff = preem_coeff

        self.window_fn = self.get_window()

    def forward(self, wavform):
        device, dtype = wavform.device, wavform.dtype
        # eps = torch.tensor(torch.finfo(torch.float).eps).to(device=device, dtype=dtype)
        eps = torch.tensor(1.1920928955078125e-07).to(device=device, dtype=dtype)
        strided_wavform = self.get_strided_wavform(wavform)

        if self.dither != 0.0:
            x = torch.max(eps, torch.rand(strided_wavform.shape, device=device, dtype=dtype))
            rand_gauss = torch.sqrt(-2 * x.log()) * torch.cos(2 * math.pi * x)
            strided_wavform = strided_wavform + rand_gauss * self.dither

        if self.remove_dc_offset:
            means = torch.mean(strided_wavform, dim=3, keepdim=True)
            strided_wavform = strided_wavform - means

        if self.raw_energy:
            signal_log_energy = self.get_log_energy(strided_wavform, eps)
        else:
            signal_log_energy = self.get_log_energy(strided_wavform, eps)

        if self.preem_coeff != 0.0:
            offset_strided_wavform = torch.nn.functional.pad(
                strided_wavform, (1, 0, 0, 0), mode='replicate'
            )
            strided_wavform = strided_wavform - self.preem_coeff * offset_strided_wavform[:, :, :, :-1]

        self.window_fn = self.window_fn.to(device=device)
        strided_wavform = strided_wavform * self.window_fn

        if self.padded_window_size != self.window_size:
            padding_right = self.padded_window_size - self.window_size
            strided_wavform = torch.nn.functional.pad(
                strided_wavform, (0, padding_right), mode='constant', value=0.0
            )

        # if not self.raw_energy:
        #     signal_log_energy = self.get_log_energy(strided_wavform, eps)
        # if version.parse(torch.__version__) < version.parse("1.8.0"):
        fft = torch.rfft(strided_wavform, 1, normalized=False, onesided=True)
        power_spectrum = fft.pow(2).sum(4)
        # else:
        #     fft = torch.fft.rfft(strided_wavform)
        #     power_spectrum = fft.pow(2).abs()
        # # spectrum = torch._C._fft.fft_rfft(strided_wavform, None, -1, None)

        return power_spectrum, signal_log_energy

    def get_window(self):
        if self.window_type == "hanning":
            return torch.hann_window(self.window_size, periodic=False)
        elif self.window_type == "hamming":
            return torch.hamming_window(self.window_size, periodic=False, alpha=0.54, beta=0.46)
        elif self.window_type == "povey":
            return torch.hann_window(self.window_size, periodic=False).pow(0.85)
        elif self.window_type == "rsectangular":
            return torch.ones(self.window_size)
        elif self.window_type == "blackman":
            a = 2 * math.pi / (self.window_size - 1)
            window_fn = torch.arange(self.window_size)
            return self.blackman_coeff - 0.5 * torch.cos(a * window_fn) + (0.5 - self.blackman_coeff) * torch.cos(
                2 * a * window_fn)
        else:
            raise Exception("Invalid window type " + self.window_type)

    def get_strided_wavform(self, wavform):
        num_samples = wavform.shape[2]
        strides = (self.window_shift * wavform.stride(2), wavform.stride(2))

        if self.snip_edges:
            if num_samples < self.window_size:
                return torch.empty((0, 0, 0), device=wavform.device, dtype=wavform.dtype)
            else:
                m = 1 + (num_samples - self.window_size) // self.window_shift
        else:
            reversed_wavform = torch.flip(wavform, [2])
            m = (num_samples + (self.window_shift // 2)) // self.window_shift
            pad = self.window_size // 2 - self.window_shift // 2
            pad_right = reversed_wavform
            if pad > 0:
                pad_left = reversed_wavform[:, :, -pad:]
                wavform = torch.cat([pad_left, wavform, pad_right], dim=2)
            else:
                wavform = torch.cat([wavform[:, :, -pad:], pad_right], dim=2)

        sizes = (m, self.window_size)
        strided_wavform = torch.zeros((wavform.shape[0], wavform.shape[1], m, self.window_size), device=wavform.device,
                                      dtype=wavform.dtype)
        for i in range(wavform.shape[0]):
            for j in range(wavform.shape[1]):
                strided_wavform[i, j] = wavform[i, j].as_strided(sizes, strides)
        return strided_wavform

    def get_log_energy(self, wavform, eps):
        log_energy = torch.max(wavform.pow(2).sum(3), eps).log()
        if self.energy_floor == 0.0:
            return log_energy
        return torch.max(log_energy,
                         torch.tensor(math.log(self.energy_floor), device=wavform.device, dtype=wavform.dtype))


class Filterbank(nn.Module):
    def __init__(self,
                 n_mels: int = 80,
                 padded_window_size: int = 512,
                 sample_rate: float = 16000,
                 low_freq: float = 20.0,
                 high_freq: float = 7600.0,
                 vtln_low: float = 100.0,
                 vtln_high: float = -500.0,
                 vtln_warp: float = 1.0,
                 use_log_fbank: bool = True
                 ):
        super().__init__()
        self.n_mels = n_mels
        self.padded_window_size = padded_window_size
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.vtln_low = vtln_low
        self.vtln_high = vtln_high
        self.vtln_warp = vtln_warp
        self.use_log_fbank = use_log_fbank

        self.filter_banks, _ = self.get_mel_banks()
        self.filter_banks = torch.nn.functional.pad(self.filter_banks, (0, 1), mode='constant', value=0)

    def forward(self, spectrum):
        device, dtype = spectrum.device, spectrum.dtype
        # eps = torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)
        eps = torch.tensor(1.1920928955078125e-07).to(device=device, dtype=dtype)

        self.filter_banks = self.filter_banks.to(device=device)

        fbanks = torch.matmul(spectrum, self.filter_banks.T)

        if self.use_log_fbank:
            fbanks = torch.max(fbanks, eps).log()

        return fbanks

    def get_mel_banks(self):
        assert self.n_mels > 3, "Must have at least 3 mels bins"
        assert self.padded_window_size % 2 == 0

        n_fft = self.padded_window_size / 2
        nyquist = 0.5 * self.sample_rate

        if self.high_freq <= 0.0:
            self.high_freq = nyquist

        assert (0.0 <= self.low_freq < nyquist) and (0.0 < self.high_freq <= nyquist) and (
                self.low_freq < self.high_freq)

        fft_bin_width = self.sample_rate / self.padded_window_size
        mel_low_freq = self.to_mel(self.low_freq)
        mel_high_freq = self.to_mel(self.high_freq)

        mel_freq_delta = (mel_high_freq - mel_low_freq) / (self.n_mels + 1)

        if self.vtln_high < 0.0:
            self.vtln_high += nyquist

        assert self.vtln_warp == 1.0 or ((self.low_freq < self.vtln_low < self.high_freq) and
                                         (0.0 < self.vtln_high < self.high_freq) and (self.vtln_low < self.vtln_high))

        bin = torch.arange(self.n_mels).unsqueeze(1)
        left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
        center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
        right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

        if self.vtln_warp != 1.0:
            left_mel = self.vtln_warp_mel_freq(left_mel)
            center_mel = self.vtln_warp_mel_freq(center_mel)
            right_mel = self.vtln_warp_mel_freq(right_mel)

        center_freqs = self.to_hz(center_mel)
        # size(1, num_fft_bins)
        mel = self.to_mel(fft_bin_width * torch.arange(n_fft)).unsqueeze(0)

        # size (num_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)

        if self.vtln_warp == 1.0:
            bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
        else:
            bins = torch.zeros_like(up_slope)
            up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)  # left_mel < mel <= center_mel
            down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)  # center_mel < mel < right_mel
            bins[up_idx] = up_slope[up_idx]
            bins[down_idx] = down_slope[down_idx]

        return bins, center_freqs

    def to_mel(self, hz):
        if isinstance(hz, float):
            return 1127.0 * math.log(1.0 + hz / 700.0)
        elif isinstance(hz, torch.Tensor):
            return 1127.0 * (1.0 + hz / 700.0).log()
        else:
            raise Exception("unsupported type " + type(hz))

    def to_hz(self, mel):
        if isinstance(mel, float):
            return 700.0 * (math.exp(mel / 1127.0) - 1.0)
        elif isinstance(mel, torch.Tensor):
            return 700.0 * ((mel / 1127.0).exp() - 1.0)
        else:
            raise Exception("unsupported type " + type(mel))

    def vtln_warp_freq(self, mel):
        assert self.vtln_low > self.low_freq, 'be sure to set the vtln_low option higher than low_freq'
        assert self.vtln_high_cutoff < self.high_freq, 'be sure to set the vtln_high option lower than high_freq [or negative]'
        l = self.vtln_low * max(1.0, self.vtln_warp)
        h = self.vtln_high * min(1.0, self.vtln_warp)
        scale = 1.0 / self.vtln_warp
        Fl = scale * l  # F(l)
        Fh = scale * h  # F(h)
        assert l > self.low_freq and h < self.high_freq
        # slope of left part of the 3-piece linear function
        scale_left = (Fl - self.low_freq) / (l - self.low_freq)
        # [slope of center part is just "scale"]

        # slope of right part of the 3-piece linear function
        scale_right = (self.high_freq - Fh) / (self.high_freq - h)

        res = torch.empty_like(mel)

        outside_low_high_freq = torch.lt(mel, self.low_freq) | torch.gt(mel,
                                                                        self.high_freq)  # freq < low_freq || freq > high_freq
        before_l = torch.lt(mel, l)  # freq < l
        before_h = torch.lt(mel, h)  # freq < h
        after_h = torch.ge(mel, h)  # freq >= h

        # order of operations matter here (since there is overlapping frequency regions)
        res[after_h] = self.high_freq + scale_right * (mel[after_h] - self.high_freq)
        res[before_h] = scale * mel[before_h]
        res[before_l] = self.low_freq + scale_left * (mel[before_l] - self.low_freq)
        res[outside_low_high_freq] = mel[outside_low_high_freq]

        return res

    def vtln_warp_mel_freq(self, mel):
        return self.to_mel(self.vtln_warp_freq(mel))


class Fbank(nn.Module):
    def __init__(self,
                 sample_rate: float = 16000,
                 frame_length: float = 25.0,
                 frame_shift: float = 10.0,
                 high_freq: float = 7600.0,
                 low_freq: float = 20.0,
                 n_mels: int = 80,
                 window_type: str = "povey",
                 vtln_low: float = 100.0,
                 vtln_high: float = -500.0,
                 vtln_warp: float = 1.0,
                 snip_edges: bool = True,
                 remove_dc_offset: bool = True,
                 subtract_mean: bool = False,
                 raw_energy: bool = True,
                 use_energy: bool = False,
                 use_log_fbank: bool = True,
                 use_power: bool = True,
                 energy_floor: float = 1.0,
                 preem_coeff: float = 0.97,
                 round_to_power_of_two: bool = True,
                 blackman_coeff: float = 0.42,
                 dither: float = 0.0
                 ):
        super().__init__()
        window_size = int(sample_rate * frame_length * 0.001)
        window_shift = int(sample_rate * frame_shift * 0.001)
        if round_to_power_of_two:
            padded_window_size = 1 if window_size == 0 else 2 ** (window_size - 1).bit_length()
        else:
            padded_window_size = window_size
        self.use_power = use_power
        self.use_energy = use_energy
        self.subtract_mean = subtract_mean
        assert window_size >= 2
        assert window_shift > 0
        assert padded_window_size % 2 == 0
        assert preem_coeff >= 0. and preem_coeff <= 1.0

        self.stft = STFT(
            window_size=window_size,
            padded_window_size=padded_window_size,
            window_shift=window_shift,
            window_type=window_type,
            blackman_coeff=blackman_coeff,
            snip_edges=snip_edges,
            raw_energy=raw_energy,
            energy_floor=energy_floor,
            dither=dither,
            remove_dc_offset=remove_dc_offset,
            preem_coeff=preem_coeff
        )

        self.filterbank = Filterbank(
            n_mels=n_mels,
            padded_window_size=padded_window_size,
            sample_rate=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            vtln_low=vtln_low,
            vtln_high=vtln_high,
            vtln_warp=vtln_warp,
            use_log_fbank=use_log_fbank
        )

    def forward(self, wavform):
        assert len(wavform.shape) in [2, 3], "input waveform must be 2 or 3 channels (batch, channel, time)"
        if len(wavform.shape) == 2:
            wavform = torch.unsqueeze(wavform, 1)

        with torch.no_grad():
            power_spectrum, signal_log_energy = self.stft(wavform)
            if not self.use_power:
                power_spectrum = power_spectrum.pow(0.5)

            fbanks = self.filterbank(power_spectrum)

            if self.use_energy:
                signal_log_energy = signal_log_energy.unsqueeze(3)
                fbanks = torch.cat([signal_log_energy, fbanks], dim=3)

            if self.subtract_mean:
                means = torch.mean(fbanks, dim=3, keepdim=True)
                fbanks = fbanks - means

            return fbanks.squeeze(1)


class FeatureNormalization(nn.Module):
    def __init__(self,
                 mean_norm=True,
                 std_norm=True,
                 requires_grad=False,
                 eps=1e-10):
        super(FeatureNormalization, self).__init__()
        self.requires_grad = requires_grad
        self.eps = eps
        self.mean_norm = mean_norm
        self.std_norm = std_norm

    def forward(self, x, lengths=None):
        return x

    def _compute_current_stats(self, x):
        """
            Args:
                x: torch.Tensor shape = (time, feat_dim)

            Not do detach in this function
        """
        if self.mean_norm:
            current_mean = torch.mean(x, dim=0)
        else:
            current_mean = torch.tensor([0.0], device=x.device)

        if self.std_norm:
            current_std = torch.std(x, dim=0)
        else:
            current_std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        current_std = torch.max(
            current_std, self.eps * torch.ones_like(current_std)
        )

        if not self.requires_grad:
            current_mean = current_mean.detach()
            current_std = current_std.detach()

        return current_mean, current_std


class SentenceFeatureNormalization(FeatureNormalization):
    def __init__(self,
                 mean_norm=True,
                 std_norm=True,
                 requires_grad=False,
                 eps=1e-10):
        super(SentenceFeatureNormalization, self).__init__(
            mean_norm=mean_norm,
            std_norm=std_norm,
            requires_grad=requires_grad,
            eps=eps
        )

    def forward(self, x, lengths: typing.Optional[torch.Tensor]=None):
        batch_num = x.shape[0]

        if lengths is None:
            lengths = torch.ones((batch_num,)) * x.shape[1]
            lengths = lengths.to(x.device)

        for i in range(batch_num):
            feat = x[i]
            length = lengths[i].int()
            cur_mean, cur_std = self._compute_current_stats(
                feat[0: length, ...]
            )
            x[i] = (x[i] - cur_mean.data) / cur_std.data

        return x


class FeatureWrapper(nn.Module):
    def __init__(self,
                 feature_module=Fbank(),
                 feature_normalization=SentenceFeatureNormalization(),
                 expands_dim=-1):
        super(FeatureWrapper, self).__init__()
        self.expand_dims = expands_dim

        self.feature_module = feature_module
        self.feature_normalization = feature_normalization

    def forward(self, x):
        x = self.feature_module(x)
        x = self.feature_normalization(x)
        if self.expand_dims >= 0:
            x = torch.unsqueeze(x, dim=self.expand_dims)
        return x
        

class EcapaTdnnSpeakerVerification(nn.Module):
    def __init__(self,
                 in_channels=80,
                 hidden_channels=512,
                 dilation_list=[1, 2, 3, 4],
                 kernel_list=[5, 3, 3, 3],
                 embedding_size=256,
                 attention_size=128,
                 scale=8,
                 se_channels=128,
                 activation=nn.ReLU,
                 use_global_concat=True):
        super(EcapaTdnnSpeakerVerification, self).__init__()
        assert len(dilation_list) == len(kernel_list)
        self.block_num = len(dilation_list)
        self.embedding_size = embedding_size

        self.conv_block = TdnnConvLayer(input_dim=in_channels,
                                        output_dim=hidden_channels,
                                        context_size=kernel_list[0],
                                        dilation=dilation_list[0],
                                        activation=activation,
                                        padding=int(dilation_list[0] * (kernel_list[0] - 1) / 2))

        self.se_res2block_list: nn.ModuleList = nn.ModuleList()
        for index in range(1, self.block_num):
            self.se_res2block_list.append(
                SERes2Block(hidden_channels,
                            hidden_channels,
                            scale=scale,
                            se_channels=se_channels,
                            kernel_size=kernel_list[index],
                            dilation=dilation_list[index],
                            activation=activation)
            )

        all_hidden_channels = (self.block_num - 1) * hidden_channels

        self.mfa = TdnnConvLayer(all_hidden_channels, all_hidden_channels, context_size=1, dilation=1, activation=activation)
        self.attentive_statistic_pooling = AttentiveStatisticsPooling(all_hidden_channels,
                                                                      attentive_channel=attention_size,
                                                                      global_context=use_global_concat)
        self.asp_bn = nn.BatchNorm1d(self.attentive_statistic_pooling.expansion * all_hidden_channels)
        self.fc = nn.Conv1d(self.attentive_statistic_pooling.expansion * all_hidden_channels,
                            out_channels=embedding_size,
                            kernel_size=1)

    def forward(self, x: torch.Tensor, lengths: typing.Optional[torch.Tensor]):

        x_s = []
        x = x.transpose(1, 2)
        x = self.conv_block(x)

        for layer in self.se_res2block_list:
            x = layer(x, lengths)
            x_s.append(x)

        x = torch.cat(x_s, dim=1)
        x = self.mfa(x)

        x = self.attentive_statistic_pooling(x)
        x = self.asp_bn(x)
        e = self.fc(x)

        embedding = e.view(-1, self.embedding_size)

        return embedding.squeeze(0)


class ModelWrapper(nn.Module):
    def __init__(self,
                 embedding_model: nn.Module = EcapaTdnnSpeakerVerification(),
                 feature_extractor: nn.Module = FeatureWrapper(),
                 ):
        super(ModelWrapper, self).__init__()

        self.feature_extractor = feature_extractor
        self.embedding_model = embedding_model

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        lengths = torch.zeros(x.shape[0], device=x.device) + 1
        x = self.feature_extractor(x)
        embedding = self.embedding_model(x, lengths)

        return embedding


if __name__ == "__main__":

    def l1_loss(train_embedding, test_embedding):
        l = min(train_embedding.shape[0],test_embedding.shape[0])
        score = torch.abs(train_embedding[:l] - test_embedding[:l])
        score = torch.mean(score)
        return score


    model = ModelWrapper()
    cpt = torch.load("/home/lycheng/workspace/corecode/Python/aslp-spknet/exp/test_ecapa/save_folder/ecapa512_grad_norm_on_voxceleb/epoch39.pth.tar", map_location='cpu')
    cpt['model'].pop('classifier.weight')
    model.load_state_dict(cpt['model'])
    model.embedding_model.se_res2block_list[0].dilated_conv.conv_list.insert(0, nn.Identity())
    model.embedding_model.se_res2block_list[1].dilated_conv.conv_list.insert(0, nn.Identity())
    model.embedding_model.se_res2block_list[2].dilated_conv.conv_list.insert(0, nn.Identity())
    model.eval()
    inputs = torch.rand(16000)
    output = model(inputs)
    print(output.shape)

    import numpy as np
    label = np.load('/home/work_nfs4_ssd/hzhao/id10270-5r0dWxy17C8-00020.npy')

    
    import scipy.io.wavfile as wf
    MAX_INT16 = np.iinfo(np.int16).max
    def read_wav(fname, normalize=True, return_rate=False):
        """
        Read wave files using scipy.io.wavfile(support multi-channel)
        """
        # samps_int16: N x C or N
        #   N: number of samples
        #   C: number of channels
        samp_rate, samps_int16 = wf.read(fname)
        # N x C => C x N
        samps = samps_int16.astype(np.float)
        # tranpose because I used to put channel axis first
        if samps.ndim != 1:
            samps = np.transpose(samps)
        # normalize like MATLAB and librosa
        if normalize:
            samps = samps / MAX_INT16
        if return_rate:
            return samp_rate, samps
        return samps

    inputs = read_wav('/home/work_nfs5_ssd/hzhao/data/voxceleb1/test/wav/id10270/5r0dWxy17C8/00020.wav')

    label = torch.Tensor(label)
    inputs = torch.Tensor(inputs)
    print(label.shape)
    # print(label)
    print(inputs.shape)
    output = model(inputs)
    print(output.shape)
    # print(output)
    print(l1_loss(output, label))

    traced_script_module = torch.jit.script(model)
    # traced_script_module.save("./jit.pt")

    output = traced_script_module(inputs)
    print(l1_loss(output, label))
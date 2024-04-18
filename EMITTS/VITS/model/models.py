import math
import torch

import torch.nn as nn

from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import functional as F

import model.attentions as attentions
import model.monotonic_align as monotonic_align
import model.modules as modules
from utils import commons


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 sin_channels=0,
                 ein_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.sin_channels = sin_channels
        self.ein_channels = ein_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                              sin_channels=sin_channels, ein_channels=ein_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, s=None, e=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, s=s, e=e)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 sin_channels=0,
                 ein_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        # add emotion
        self.sin_channels = sin_channels
        self.ein_channels = ein_channels

        self.flows = nn.ModuleList()
        # add emotion
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels,
                                                            hidden_channels,
                                                            kernel_size,
                                                            dilation_rate,
                                                            n_layers,
                                                            sin_channels=sin_channels,
                                                            ein_channels=ein_channels,
                                                            mean_only=True))
            self.flows.append(modules.Flip())

    # add emotion
    def forward(self, x, x_mask, s=None, e=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, s=s, e=e, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, s=s, e=e, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 sin_channels=0,
                 ein_channels=0):
        super().__init__()
        # it needs to be removed from future version.
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.sin_channels = sin_channels
        # add emotion
        self.ein_channels = ein_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(
                2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(
                2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)  # 192, 3, 3, 0.5
        if sin_channels != 0:
            self.scond = nn.Conv1d(sin_channels, filter_channels, 1)
        # add emotion
        if ein_channels != 0:
            self.econd = nn.Conv1d(ein_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, s=None, e=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if s is not None:
            s = torch.detach(s)
            x = x + self.scond(s)
        # add emotion
        if e is not None:
            e = torch.detach(e)
            x = x + self.econd(e)

        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(
                device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, c=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) +
                                      F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2))
                             * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, c=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2))
                            * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(
                device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, c=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class Vocoder(nn.Module):
    def __init__(self,
                 initial_channel,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 sin_channels=0,
                 ein_channels=0):
        super(Vocoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if sin_channels != 0:
            self.scond = nn.Conv1d(sin_channels, upsample_initial_channel, 1)
        # add emotion
        if ein_channels != 0:
            self.econd = nn.Conv1d(ein_channels, upsample_initial_channel, 1)

    def forward(self, x, s=None, e=None):
        x = self.conv_pre(x)
        if s is not None:
            x = x + self.scond(s)
        # add emotion
        if e is not None:
            x = x + self.econd(e)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 sin_channels=0,
                 efeature_dim=0,
                 ein_channels=0,
                 use_sdp=True,
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.efeature_dim = efeature_dim
        self.sin_channels = sin_channels
        self.ein_channels = ein_channels
        self.use_sdp = use_sdp

        if n_speakers > 0:
            self.emb_s = nn.Embedding(n_speakers, sin_channels)
        
        if efeature_dim > 0:
            self.emb_e = nn.Linear(efeature_dim, ein_channels)

        self.textEncoder = TextEncoder(n_vocab,
                                       inter_channels,
                                       hidden_channels,
                                       filter_channels,
                                       n_heads,
                                       n_layers,
                                       kernel_size,
                                       p_dropout)
        # posterir Encoder w/o emotion
        # self.posteriorEncoder = PosteriorEncoder(spec_channels,
        #                                          inter_channels,
        #                                          hidden_channels,
        #                                          kernel_size=5,
        #                                          dilation_rate=1,
        #                                          n_layers=16,
        #                                          sin_channels=sin_channels,
        #                                          ein_channels=ein_channels)
        self.posteriorEncoder = PosteriorEncoder(spec_channels,
                                            inter_channels,
                                            hidden_channels,
                                            kernel_size=5,
                                            dilation_rate=1,
                                            n_layers=16,
                                            sin_channels=sin_channels,
                                            ein_channels=0)
        self.flow = ResidualCouplingBlock(inter_channels,
                                          hidden_channels,
                                          5, 1, 4,
                                          sin_channels=sin_channels,
                                          ein_channels=ein_channels)
        self.stochasticDurationPredictor = StochasticDurationPredictor(hidden_channels,
                                                                       filter_channels=192,
                                                                       kernel_size=3,
                                                                       p_dropout=0.5,
                                                                       n_flows=4,
                                                                       sin_channels=sin_channels,
                                                                       ein_channels=ein_channels)
        # vocoder w/o emotion
        # self.vocoder = Vocoder(inter_channels,
        #                        resblock,
        #                        resblock_kernel_sizes,
        #                        resblock_dilation_sizes,
        #                        upsample_rates,
        #                        upsample_initial_channel,
        #                        upsample_kernel_sizes,
        #                        sin_channels=sin_channels,
        #                        ein_channels=ein_channels)
        self.vocoder = Vocoder(inter_channels,
                               resblock,
                               resblock_kernel_sizes,
                               resblock_dilation_sizes,
                               upsample_rates,
                               upsample_initial_channel,
                               upsample_kernel_sizes,
                               sin_channels=sin_channels,
                               ein_channels=0)

    def forward(self, x, x_lengths, y, y_lengths, sid=None, efeature=None):
        x, m_p, logs_p, x_mask = self.textEncoder(x, x_lengths)
        if self.n_speakers > 0:
            s = self.emb_s(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            s = None

        # add emotion
        if self.efeature_dim > 0:
            e = self.emb_e(efeature).unsqueeze(-1)  # [b, ein_channels, 1]
        else:
            e = None

        z, m_q, logs_q, y_mask = self.posteriorEncoder(y, y_lengths, s=s, e=e)
        # posterir Encoder w/o emotion
        # z, m_q, logs_q, y_mask = self.posteriorEncoder(y, y_lengths, s=s, e=None)
            

        z_p = self.flow(z, y_mask, s=s, e=e)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_t]
            # [b, 1, t_t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 *
                                  math.pi) - logs_p, [1], keepdim=True)
            # [b, t_s, d] x [b, d, t_t] = [b, t_s, t_t]
            neg_cent2 = torch.matmul(-0.5 *
                                     (z_p ** 2).transpose(1, 2), s_p_sq_r)
            # [b, t_s, d] x [b, d, t_t] = [b, t_s, t_t]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) *
                                  s_p_sq_r, [1], keepdim=True)  # [b, 1, t_t]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(
                1)).unsqueeze(1).detach()  # [b, 1, t_s, t_t]

        w = attn.sum(2)
        l_length = self.stochasticDurationPredictor(
            x, x_mask, w,  s=s, e=e, reverse=False)
        l_length = l_length / torch.sum(x_mask)  # [b]

        # expand prior
        m_p = torch.matmul(attn.squeeze(
            1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(
            1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size)

        output = self.vocoder(z_slice, s=s, e=e)
        # vocoder w/o emotion
        # output = self.vocoder(z_slice, s=s, e=None)
        return output, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, efeature=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.textEncoder(x, x_lengths)
        if self.n_speakers > 0:
            s = self.emb_s(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            s = None

        # add emotion
        if self.efeature_dim > 0:
            e = self.emb_e(efeature).unsqueeze(-1)  # [b, efeature_dim, 1]
        else:
            e = None

        logw = self.stochasticDurationPredictor(
            x, x_mask, s=s, e=e, reverse=True, noise_scale=noise_scale)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)  # rounded up
        y_lengths = torch.clamp_min(
            torch.sum(w_ceil, [1, 2]), 1).long()  # make sure y_lengths at least 1
        y_mask = torch.unsqueeze(commons.sequence_mask(
            y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, s=s, e=e, reverse=True)

        # vocoder w/o emotion
        # output = self.vocoder((z * y_mask)[:, :, :max_len], s=s, e=e)
        output = self.vocoder((z * y_mask)[:, :, :max_len], s=s, e=None)
        return output, attn, y_mask, (z, z_p, m_p, logs_p)


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                   padding=(commons.get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                   padding=(commons.get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                   padding=(commons.get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                   padding=(commons.get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,
                   padding=(commons.get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + \
            [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
             for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

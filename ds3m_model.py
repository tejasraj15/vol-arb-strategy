"""
DS3M (Deep Switching State Space Model) - Core Model
Based on Xu, Peng & Chen (2026), International Journal of Forecasting
GitHub: https://github.com/Sherry-Xu/Deep-Switching-State-Space-Model/

This is the model architecture. It takes in time series data and learns:
  - Which regime the market is in (d_t)
  - The hidden driving factors (z_t)
  - A forecast for the next observation (y_t)
"""

import math
import torch
import torch.nn as nn
import numpy as np


class DS3M(nn.Module):

    def __init__(self, x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device):
        """
        Args:
            x_dim:    dimension of input features (e.g. lagged vol, IV, VIX)
            y_dim:    dimension of target (1 if just realized vol)
            h_dim:    GRU hidden state size (memory capacity)
            z_dim:    continuous latent variable dimension
            d_dim:    number of regimes (K=2 for calm/stress)
            n_layers: number of GRU layers
            device:   'cuda' or 'cpu'
        """
        super(DS3M, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.n_layers = n_layers
        self.device = device

        # ---- Transition Matrix Prior ----
        self.Transition_initial = (
            torch.eye(d_dim, device=device) * (1 - 0.05 * d_dim)
            + torch.ones((d_dim, d_dim), device=device) * 0.05
        )
        self.dprior = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Softmax(dim=1)
        )

        # ---- Per-regime networks ----
        # Generative model (used in training AND forecasting)
        self.ztrans = nn.ModuleList()
        self.ztrans_mean = nn.ModuleList()
        self.ztrans_std = nn.ModuleList()

        self.yemit = nn.ModuleList()
        self.yemit_mean = nn.ModuleList()
        self.yemit_std = nn.ModuleList()

        # Inference network (training only)
        self.dpost = nn.ModuleList()
        self.zpost = nn.ModuleList()
        self.zpost_mean = nn.ModuleList()
        self.zpost_std = nn.ModuleList()

        for k in range(d_dim):
            # d posterior
            self.dpost.append(nn.Sequential(
                nn.Linear(h_dim, d_dim), nn.Softmax(dim=1)))

            # z posterior
            self.zpost.append(nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim), nn.ReLU(),
                nn.Linear(z_dim, z_dim), nn.ReLU()))
            self.zpost_mean.append(nn.Linear(z_dim, z_dim))
            self.zpost_std.append(nn.Sequential(
                nn.Linear(z_dim, z_dim), nn.Softplus()))

            # z transition (generative)
            self.ztrans.append(nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim), nn.ReLU(),
                nn.Linear(z_dim, z_dim), nn.ReLU()))
            self.ztrans_mean.append(nn.Linear(z_dim, z_dim))
            self.ztrans_std.append(nn.Sequential(
                nn.Linear(z_dim, z_dim), nn.Softplus()))

            # y emission (generative)
            self.yemit.append(nn.Sequential(
                nn.Linear(z_dim + h_dim, y_dim), nn.ReLU(),
                nn.Linear(y_dim, y_dim), nn.ReLU()))
            self.yemit_mean.append(nn.Linear(y_dim, y_dim))
            self.yemit_std.append(nn.Sequential(
                nn.Linear(y_dim, y_dim), nn.Softplus()))

        # Forward GRU: h_t = f_h(h_{t-1}, x_t)
        self.rnn_forward = nn.GRU(x_dim, h_dim, n_layers)
        # Backward RNN: A_t = g(A_{t+1}, [y_t, h_t])
        self.rnn_backward = nn.GRU(y_dim + h_dim, h_dim, n_layers)

    def get_transition_matrix(self):
        T = self.dprior(self.Transition_initial) / 2 + torch.eye(self.d_dim, device=self.device) / 2
        return T

    def forward(self, x, y):
        """
        Training forward pass. Uses both forward and backward RNN.
        x: (seq_len, batch, x_dim)
        y: (seq_len, batch, y_dim)
        """
        T, B = x.size(0), x.size(1)
        Trans = self.get_transition_matrix()

        # Storage
        all_d_post, all_d_plot, all_d_oh = [], [], []
        all_z_sampled, all_z_post_mean, all_z_post_std = [], [], []
        all_y_mean, all_y_std = [], []
        kld_g, kld_c, nll = 0, 0, 0

        # Init
        h0 = torch.zeros((self.n_layers, B, self.h_dim), device=self.device)
        A0 = torch.zeros((self.n_layers, B, self.h_dim), device=self.device)

        d0_s = torch.distributions.Categorical(
            torch.ones(self.d_dim) / self.d_dim).sample((B,)).long()
        d0_oh = torch.eye(self.d_dim, device=self.device)[d0_s]

        all_d_post.append(torch.ones((B, self.d_dim), device=self.device) / self.d_dim)
        all_d_plot.append(d0_s.reshape(-1, 1).to(self.device))
        all_d_oh.append(d0_oh)

        z0 = torch.zeros((B, self.z_dim), device=self.device)
        all_z_sampled.append(z0)
        all_z_post_mean.append(z0)
        all_z_post_std.append(z0)

        # Forward GRU
        h_all, _ = self.rnn_forward(x, h0)

        # Backward RNN
        yh = torch.cat([y, h_all], dim=2)
        A_all, _ = self.rnn_backward(torch.flip(yh, [0]), A0)

        for t in range(T):
            h_t = h_all[t]
            A_t = A_all[T - t - 1]

            # d posterior
            d_post = torch.zeros((B, self.d_dim), device=self.device)
            d_post_k = []
            for k in range(self.d_dim):
                dk = self.dpost[k](A_t)
                d_post_k.append(dk)
                d_post += dk * all_d_oh[t][:, k:(k+1)]
            all_d_post.append(d_post)

            d_samp = torch.distributions.Categorical(d_post).sample().long().to(self.device)
            all_d_plot.append(d_samp.reshape(-1, 1))
            all_d_oh.append(torch.eye(self.d_dim, device=self.device)[d_samp])

            # z prior and posterior
            zpr_m_k, zpr_s_k, zpo_m_k, zpo_s_k = [], [], [], []
            z_pr_m = torch.zeros((B, self.z_dim), device=self.device)
            z_pr_s = torch.zeros((B, self.z_dim), device=self.device)
            z_po_m = torch.zeros((B, self.z_dim), device=self.device)
            z_po_s = torch.zeros((B, self.z_dim), device=self.device)

            for k in range(self.d_dim):
                w = all_d_oh[t+1][:, k:(k+1)]

                zp = self.ztrans[k](torch.cat([h_t, all_z_sampled[t]], 1))
                pm, ps = self.ztrans_mean[k](zp), self.ztrans_std[k](zp)
                zpr_m_k.append(pm); zpr_s_k.append(ps)
                z_pr_m += pm * w; z_pr_s += ps * w

                zq = self.zpost[k](torch.cat([A_t, all_z_sampled[t]], 1))
                qm, qs = self.zpost_mean[k](zq), self.zpost_std[k](zq)
                zpo_m_k.append(qm); zpo_s_k.append(qs)
                z_po_m += qm * w; z_po_s += qs * w

            all_z_post_mean.append(z_po_m)
            all_z_post_std.append(z_po_s)

            # Reparameterization trick
            z_t = z_po_m + torch.randn_like(z_po_s) * z_po_s
            all_z_sampled.append(z_t)

            # y emission
            ym_k, ys_k = [], []
            y_m = torch.zeros((B, self.y_dim), device=self.device)
            y_s = torch.zeros((B, self.y_dim), device=self.device)

            for k in range(self.d_dim):
                w = all_d_oh[t+1][:, k:(k+1)]
                ye = self.yemit[k](torch.cat([h_t, all_z_sampled[t+1]], 1))
                em, es = self.yemit_mean[k](ye), self.yemit_std[k](ye)
                ym_k.append(em); ys_k.append(es)
                y_m += em * w; y_s += es * w

            all_y_mean.append(y_m)
            all_y_std.append(y_s)

            # ELBO losses (equation 21)
            for k in range(self.d_dim):
                kld_g += torch.sum(
                    self._kld_gauss(zpo_m_k[k], zpo_s_k[k], zpr_m_k[k], zpr_s_k[k])
                    * d_post[:, k:(k+1)])
                kld_c += torch.sum(
                    self._kld_cat(d_post_k[k], Trans[k:(k+1), :])
                    * all_d_post[-2][:, k])
                nll += torch.sum(
                    self._nll_gauss(ym_k[k], ys_k[k], y[t])
                    * d_post[:, k:(k+1)])

        return {
            'kld_gaussian': kld_g, 'kld_category': kld_c, 'nll': nll,
            'd_posterior': all_d_post, 'd_sampled': all_d_plot,
            'z_sampled': all_z_sampled, 'y_mean': all_y_mean, 'y_std': all_y_std,
            'd_onehot': all_d_oh,
        }

    @torch.no_grad()
    def forecast(self, x, y, steps=1, n_samples=100):
        """
        Multi-step Monte Carlo forecasting (equation 22).
        Call this at trading time.
        """
        self.eval()
        B = x.size(1)
        Trans = self.get_transition_matrix()

        all_fc, all_rp, all_rs = [], [], []

        for s in range(n_samples):
            out = self.forward(x, y)
            z_cur = out['z_sampled'][-1]
            d_cur = out['d_onehot'][-1]

            h0 = torch.zeros((self.n_layers, B, self.h_dim), device=self.device)
            _, h_st = self.rnn_forward(x, h0)

            fc_s, rp_s, rs_s = [], [], []
            # Keep recursive GRU inputs in x-space (x_dim), not y-space (y_dim).
            # We update only the lagged-RV feature (index 0) with the latest forecast
            # and carry the remaining features forward as a simple persistence baseline.
            x_state = x[-1, :, :].clone()
            x_next = x_state.unsqueeze(0)

            for t in range(steps):
                _, h_st = self.rnn_forward(x_next, h_st)
                h_t = h_st.squeeze(0)

                d_prior = torch.mm(d_cur, Trans)
                d_samp = torch.distributions.Categorical(d_prior).sample().long()
                d_oh = torch.eye(self.d_dim, device=self.device)[d_samp]

                rp_s.append(d_prior.cpu().numpy())
                rs_s.append(d_samp.cpu().numpy())

                z_m = torch.zeros((B, self.z_dim), device=self.device)
                z_s = torch.zeros((B, self.z_dim), device=self.device)
                for k in range(self.d_dim):
                    zp = self.ztrans[k](torch.cat([h_t, z_cur], 1))
                    z_m += self.ztrans_mean[k](zp) * d_oh[:, k:(k+1)]
                    z_s += self.ztrans_std[k](zp) * d_oh[:, k:(k+1)]
                z_cur = torch.distributions.Normal(z_m, z_s).sample()

                y_m = torch.zeros((B, self.y_dim), device=self.device)
                y_s = torch.zeros((B, self.y_dim), device=self.device)
                for k in range(self.d_dim):
                    ye = self.yemit[k](torch.cat([h_t, z_cur], 1))
                    y_m += self.yemit_mean[k](ye) * d_oh[:, k:(k+1)]
                    y_s += self.yemit_std[k](ye) * d_oh[:, k:(k+1)]

                y_t = torch.distributions.Normal(y_m, y_s).sample()
                fc_s.append(y_t.cpu().numpy())
                if self.x_dim > 0:
                    x_state[:, 0] = y_t.squeeze(-1)
                x_next = x_state.unsqueeze(0)
                d_cur = d_oh

            all_fc.append(fc_s)
            all_rp.append(rp_s)
            all_rs.append(rs_s)

        fc = np.array(all_fc)
        rp = np.array(all_rp)

        return {
            'vol_forecast_mean': np.mean(fc, axis=0),
            'vol_forecast_std': np.std(fc, axis=0),
            'vol_forecast_q05': np.percentile(fc, 5, axis=0),
            'vol_forecast_q95': np.percentile(fc, 95, axis=0),
            'regime_probs': np.mean(rp, axis=0),
            'transition_matrix': Trans.cpu().numpy(),
        }

    def _kld_gauss(self, mq, sq, mp, sp):
        return 0.5 * (2*torch.log(sp) - 2*torch.log(sq) + (sq**2 + (mq-mp)**2)/sp**2 - 1)

    def _kld_cat(self, q, p):
        return torch.sum(q * torch.log(q / p), dim=1)

    def _nll_gauss(self, m, s, x):
        return 0.5*math.log(2*math.pi) + torch.log(s) + (x-m)**2/(2*s**2)

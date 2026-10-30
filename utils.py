"""Shared transport, simulation, metric, and plotting utilities for the SUDO experiments.
   Adopted from USB.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import ot as pot
import math
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from pathlib import Path

# Learn velocity, score, and growth fields along the stochastic bridge.
class Possion_Bridge(nn.Module):
    def __init__(self, dims, nu, activation = 'leakyrelu'):
        super(Possion_Bridge, self).__init__()
        self.dims = dims
        self.nu = nu
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        blocks_v = []
        for i in range(len(dims)-1):
            blocks_v.append(nn.Linear(dims[i], dims[i+1]))
            blocks_v.append(self.activation)
        blocks_v.append(nn.Linear(dims[-1], dims[0] - 1))
        self.v_net = nn.Sequential(*blocks_v)

        blocks_s = []
        for i in range(len(dims)-1):
            blocks_s.append(nn.Linear(dims[i], dims[i+1]))
            blocks_s.append(self.activation)
        blocks_s.append(nn.Linear(dims[-1], 1))
        self.s_net = nn.Sequential(*blocks_s)

        blocks_k = []
        for i in range(len(dims)-1):
            blocks_k.append(nn.Linear(dims[i], dims[i+1]))
            blocks_k.append(self.activation)
        blocks_k.append(nn.Linear(dims[-1], 1))
        self.k_net = nn.Sequential(*blocks_k)

    def forward(self, x, t):


        v = self.v_net(torch.concatenate([x, t], dim=-1))
        s = self.s_net(torch.concatenate([x, t], dim=-1))
        k = self.k_net(torch.concatenate([x, t], dim=-1))
        return v, s, k

    def sample_comditional_flow(self, x0, x1, m0, m1, t):


        mean = t * x1 + (1-t) * x0
        sigma = torch.sqrt(t * (1-t)) * self.nu

        eps = torch.randn_like(mean)

        xts = mean + sigma*eps

        vts = (1-2*t)/(t*(1-t)+1e-8)*(xts-mean)+(x1-x0)

        mts = t * m1 + (1-t) * m0
        kts = (m1 - mts)/(1-t+1e-8)

        return xts, vts, kts, eps, mts

    def sample(self, x, t, N = 1000):
        if x is None:
            x = torch.randn(self.dims[0] - 1)
        if isinstance(t, float):
            t = torch.tensor([t])
        dt = t / N

        m = torch.ones_like(t)
        for i in range(N):
            ipt = torch.cat([x, i * t / N], dim=-1)
            v, s, k = self.forward(ipt)
            u = v - s
            x = x + u * dt + self.nu * torch.sqrt(dt) * torch.randn(x.shape)
            m += k * dt

        return x, m


# Simulate particle positions and log-masses through time.
class SDE:
    def __init__(self, model, nu, mode = 1, positive=True):
        self.drift = model.v_net
        self.score = model.s_net
        self.rate = model.k_net
        self.nu = nu
        self.mode = mode
        self.positive = positive

    def trajectory(self,x,m,delta,N=100,T=1,I=None):
        xs = [x.detach().numpy().copy()]
        ms = [m.detach().numpy().copy()]

        if I:
            dt = (I[1]-I[0])/N
            ts = torch.linspace(I[0],I[1]-dt,N)
        else:
            dt = T/N
            ts = torch.linspace(0,T-dt,N)
        action = 0

        if self.mode == 1:
            print('Continuous inference')
            for tt in ts:
                t = tt*torch.ones([x.size(0),1])
                v = self.drift(torch.concatenate([x,t],dim=-1))
                s = self.score(torch.concatenate([x,t],dim=-1))
                k = self.rate(torch.concatenate([x,t],dim=-1))


                action += 0.5*torch.sum(m*(v+s)**2)*dt + 0.5*delta*delta*torch.sum(k**2/(m+1e-2))*dt


                x += (v + s)*dt + self.nu*np.sqrt(dt)*torch.randn([x.size(0),1])
                m += k*dt
                if self.positive:
                    m = torch.max(torch.zeros_like(m), m)

                xs.append(x.detach().numpy().copy())
                ms.append(m.detach().numpy().copy())


        elif self.mode == 2:
            print('Poisson inference')
            for tt in ts:
                t = tt*torch.ones([x.size(0),1])
                v = self.drift(torch.concatenate([x,t],dim=-1))
                s = self.score(torch.concatenate([x,t],dim=-1))
                k = self.rate(torch.concatenate([x,t],dim=-1))


                action += 0.5*torch.sum(m*(v+s)**2)*dt + 0.5*delta*delta*torch.sum(k**2/(m+1e-2))*dt


                x += (v + s)*dt + self.nu*np.sqrt(dt)*torch.randn([x.size(0),1])

                m += (torch.rand([x.size(0),1])<torch.abs(50*k*dt)) * (k.sign())/50
                if self.positive:
                    m = torch.max(torch.zeros_like(m), m)

                xs.append(x.detach().numpy().copy())
                ms.append(m.detach().numpy().copy())

        elif self.mode == 3:


            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])


            from collections import defaultdict
            genealogy = defaultdict(lambda: {'t': [], 'x': []})
            current_ids = torch.arange(x.size(0))
            next_unique_id = x.size(0)


            t_start = I[0] if I else 0
            for i, pid in enumerate(current_ids.flatten().cpu().numpy()):
                genealogy[pid]['t'].append(t_start)
                genealogy[pid]['x'].append(x[i].detach().numpy().copy())

            curr_x = x

            for tt in ts:
                if curr_x.size(0) == 0:
                    break


                N_curr = curr_x.shape[0]


                t_tensor = tt * torch.ones([N_curr, 1]).to(curr_x.device)


                if curr_x.ndim != t_tensor.ndim:


                    curr_x = curr_x.view(N_curr, -1)

                net_in = torch.concatenate([curr_x, t_tensor], dim=-1)


                v = self.drift(net_in)
                s = self.score(net_in)
                k = self.rate(net_in)


                noise = torch.randn_like(curr_x) * self.nu * np.sqrt(dt)
                dx = (v + s) * dt + noise


                curr_x = curr_x + dx


                prob_event = torch.abs(k) * dt
                prob_event = torch.clamp(prob_event, max=1.0)

                rand_u = torch.rand_like(prob_event)
                is_event = rand_u < prob_event


                split_mask = is_event & (k > 0)

                die_mask = is_event & (k < 0)
                keep_mask = ~die_mask.squeeze()


                if keep_mask.ndim > 1: keep_mask = keep_mask.squeeze()

                x_survivors = curr_x[keep_mask]
                ids_survivors = current_ids[keep_mask]

                survivor_is_splitting = split_mask[keep_mask].squeeze()
                if survivor_is_splitting.ndim == 0:
                    survivor_is_splitting = survivor_is_splitting.unsqueeze(0)

                if survivor_is_splitting.sum() > 0:
                    x_new_born = x_survivors[survivor_is_splitting]
                    num_new = x_new_born.size(0)

                    new_ids = torch.arange(next_unique_id, next_unique_id + num_new)
                    next_unique_id += num_new


                    curr_time = (tt + dt).item()
                    for i, nid in enumerate(new_ids.flatten().cpu().numpy()):
                        genealogy[nid]['t'].append(curr_time)
                        genealogy[nid]['x'].append(x_new_born[i].detach().numpy().copy())


                    curr_x = torch.cat([x_survivors, x_new_born], dim=0)


                    if ids_survivors.device != new_ids.device:
                        new_ids = new_ids.to(ids_survivors.device)

                    current_ids = torch.cat([ids_survivors.flatten(), new_ids.flatten()], dim=0)

                else:
                    curr_x = x_survivors
                    current_ids = ids_survivors


                curr_time = (tt + dt).item()
                valid_ids = current_ids.flatten().cpu().numpy()

                for i, pid in enumerate(valid_ids):
                    genealogy[pid]['t'].append(curr_time)
                    genealogy[pid]['x'].append(curr_x[i].flatten().detach().numpy().copy())


            return genealogy, None, 0

        else:
            raise NotImplementedError('only 1=\'Continuous\', 2=\'Poisson\', 3=\'Branching\' are valid!!')
        return np.array(xs), np.array(ms).squeeze(), action


def ma(data):


    if len(data) == 0:
        return []

    ma = [data[0]]
    for t in range(1, len(data)):
        ma_t = ma[-1] + (data[t]-ma[-1])/(t+1)
        ma.append(ma_t)

    return ma

# Compute the empirical Wasserstein distance.
def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:

    ot_fn = pot.emd2
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

# Compute Wasserstein distance after normalizing particle masses.
def wasserstein_with_weights(x0: torch.Tensor, m0, x1: torch.Tensor, m1, method="exact", reg=0.05, power=1) -> float:

    ot_fn = pot.emd2
    a, b = m0/m0.sum(), m1/m1.sum()

    x0 = x0.reshape(x0.shape[0], -1)
    x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1, p=2)
    if power == 2:
        M = M ** 2
    dist = ot_fn(a, b, M.cpu().numpy(), numItermax=1e7)
    return math.sqrt(dist) if power == 2 else dist

# Solve adjacent unbalanced transport problems for the observed snapshots.
def compute_uot_plans(X, t_train, delta=1, use_mini_batch_uot=False, chunk_size=2000, draw=False, cuda=False):


    uot_plans = []
    gamma0_plans = []
    gamma1_plans = []


    total_action = 0
    if not cuda:
        for i in tqdm(range(len(t_train)-1), desc="Computing UOT plans..."):
            X_source = X[i]
            X_target = X[i+1]
            n_source, n_target = X_source.shape[0], X_target.shape[0]
            norm_2_dist = pot.dist(X_source, X_target, metric='euclidean')

            cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi/2))**2
            cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))


            if not use_mini_batch_uot:
                a = np.ones(n_source)
                b = np.ones(n_target)
                G = pot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=[1.0,1.0])
                total_action += 2*(delta**2)*pot.unbalanced.mm_unbalanced2(a, b, cost_matrix, reg_m=[1.0,1.0], returnCost='total')
            else:

                group_number = n_source // chunk_size + 1

                a = np.ones(n_source)
                b = np.ones(n_target)
                G = np.zeros((n_source, n_target))


                source_perm = np.arange(n_source)
                np.random.shuffle(source_perm)
                target_perm = np.arange(n_target)
                np.random.shuffle(target_perm)


                source_indices = np.array_split(source_perm, group_number)
                target_indices = np.array_split(target_perm, group_number)


                for src_idx,tgt_idx in zip(source_indices,target_indices):
                    sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                    sub_a = a[src_idx]
                    sub_b = b[tgt_idx]
                    G_sub = pot.unbalanced.mm_unbalanced(sub_a, sub_b, sub_cost_matrix, reg_m=[1.0,1.0])
                    G[np.ix_(src_idx, tgt_idx)] = G_sub


            gamma0_plan = ((a / G.sum(1))[:, None]) * G
            gamma1_plan = (b/G.sum(0))*G


            uot_plans.append(G)
            gamma0_plans.append(gamma0_plan)
            gamma1_plans.append(gamma1_plan)

            print(((gamma1_plans[i]- gamma0_plans[i])<0).any())


    else:
        for i in tqdm(range(len(t_train)-1), desc="Computing UOT plans..."):
            X_source = X[i]
            X_target = X[i+1]
            n_source, n_target = X_source.shape[0], X_target.shape[0]
            norm_2_dist = pot.dist(X_source, X_target, metric='euclidean')
            device = 'cuda'

            cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi/2))**2
            cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

            a = np.ones(n_source)
            b = np.ones(n_target)


            if not use_mini_batch_uot:
                a_cuda = torch.from_numpy(a).to(device)
                b_cuda = torch.from_numpy(b).to(device)
                cost_matrix_cuda = torch.from_numpy(cost_matrix).to(device)
                G = pot.unbalanced.mm_unbalanced(a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0,1.0])
                G = G.cpu().numpy()
                total_action += 2*(delta**2)*(pot.unbalanced.mm_unbalanced2(a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0,1.0], returnCost='total')).cpu().numpy()

            else:
                group_number = n_source // chunk_size + 1
                G = np.zeros((n_source, n_target))
                source_perm = np.arange(n_source)
                np.random.shuffle(source_perm)
                target_perm = np.arange(n_target)
                np.random.shuffle(target_perm)
                source_indices_groups = np.array_split(source_perm, group_number)
                target_indices_groups = np.array_split(target_perm, group_number)

                gamma0_sub_plans = []
                for src_idx, tgt_idx in zip(source_indices_groups, target_indices_groups):
                    sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                    sub_a = a[src_idx]
                    sub_b = b[tgt_idx]
                    sub_a_cuda = torch.from_numpy(sub_a).to(device)
                    sub_b_cuda = torch.from_numpy(sub_b).to(device)
                    sub_cost_matrix = torch.from_numpy(sub_cost_matrix).to(device)
                    G_sub = pot.unbalanced.mm_unbalanced(sub_a_cuda, sub_b_cuda, sub_cost_matrix, reg_m=[1.0,1.0])
                    G_sub = G_sub.cpu().numpy()
                    G[np.ix_(src_idx, tgt_idx)] = G_sub

                    G_sub_sum_1 = G_sub.sum(1)
                    gamma0_sub = ((sub_a / (G_sub_sum_1 + 1e-12))[:, None]) * G_sub
                    gamma0_sub_plans.append(gamma0_sub.astype(np.float32))


            gamma0_plan = ((a / G.sum(1))[:, None]) * G
            gamma1_plan = (b/G.sum(0))*G


            uot_plans.append(G)
            gamma0_plans.append(gamma0_plan)
            gamma1_plans.append(gamma1_plan)

            print(((gamma1_plans[i]- gamma0_plans[i])<0).any())


        if draw:
            source_pred = G.sum(1)
            tar_pred = G.sum(0)

            fig = plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.plot(a, label = f'source_true_{i}')
            plt.plot(source_pred, label = f'source_pred_{i}')
            plt.legend()

            plt.subplot(132)
            plt.plot(b, label = f'target_true_{i+1}')
            plt.plot(tar_pred, label = f'target_pred_{i+1}')
            plt.legend()

            plt.subplot(133)
            plt.scatter(X_source[:,0],X_source[:,1],s=source_pred*10, alpha=0.5)
            plt.show()

    print('true_action', total_action)

    return uot_plans, gamma0_plans, gamma1_plans, total_action


def sample_map(pi: np.ndarray, batch_size: int = 256, replace: bool = True):


    row_sums = pi.sum(axis=1)
    total_sum = row_sums.sum()
    row_probs = row_sums / total_sum


    i_samples = np.random.choice(pi.shape[0], p=row_probs, size=batch_size, replace=replace)


    j_samples = np.zeros(batch_size, dtype=int)
    for idx, i in enumerate(i_samples):

        row_p = pi[i] / row_sums[i]
        j_samples[idx] = np.random.choice(pi.shape[1], p=row_p)

    return i_samples, j_samples


# Sample endpoint pairs according to a transport plan.
def sample_from_ot_plan(ot_plan: np.ndarray, x0: torch.Tensor, x1: torch.Tensor, batch_size: int = 256):


    i, j = sample_map(ot_plan, batch_size, replace=True)
    return x0[i], x1[j], i, j


# Plot predicted growth values on the observed points.
def plot_g_values(Xs, mass_ratio, model, output_file='g_plot.pdf'):


    data_by_time = {}
    n = Xs[0].shape[1]

    for time in range(len(Xs)):
        subset = Xs[time]
        column_names = [f'x{i}' for i in range(1, n + 1)]
        tensors = torch.tensor(subset, dtype=torch.float32)

        with torch.no_grad():
            t = time*torch.ones((Xs[time].shape[0],1))
            _, _, k = model.forward(tensors, t)
            k = k.detach().cpu().numpy()
            g = k/mass_ratio[time]

        data_by_time[time] = {'data': subset, 'g_values': g}

    all_g_values = np.concatenate([content['g_values'] for content in data_by_time.values()])

    vmax_value = np.percentile(all_g_values, 100)
    vmin_value = np.percentile(all_g_values, 0)

    norm = plt.Normalize(vmin=vmin_value, vmax=vmax_value, clip=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for time, content in data_by_time.items():
        subset = content['data']
        g_values = content['g_values']
        x = subset[:,0]
        y = subset[:,1]

        colors = plt.cm.plasma(norm(g_values))

        ax.scatter(x, y, color=colors, label=f'Time {time}', alpha=0.7, marker='o')

    ax.set_xlabel('$X_1$',fontsize=16)
    ax.set_ylabel('$X_2$',fontsize=16)

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array(all_g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted rate',fontsize=16)

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{norm(x):.2f}'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file,bbox_inches='tight',transparent=True)
    plt.close()


# Plot the learned growth field at each observation time.
def plot_k_values(Xs, model, output_file='k_plot.pdf'):


    data_by_time = {}
    n = Xs[0].shape[1]

    for time in range(len(Xs)):
        subset = Xs[time]
        column_names = [f'x{i}' for i in range(1, n + 1)]
        tensors = torch.tensor(subset, dtype=torch.float32)

        with torch.no_grad():
            t = time*torch.ones((Xs[time].shape[0],1))
            _, _, k = model.forward(tensors, t)

        data_by_time[time] = {'data': subset, 'k_values': k.detach().cpu().numpy()}

    all_k_values = np.concatenate([content['k_values'] for content in data_by_time.values()])

    vmax_value = np.percentile(all_k_values, 100)
    vmin_value = np.percentile(all_k_values, 0)

    norm = plt.Normalize(vmin=vmin_value, vmax=vmax_value, clip=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for time, content in data_by_time.items():
        subset = content['data']
        k_values = content['k_values']
        x = subset[:,0]
        y = subset[:,1]

        colors = plt.cm.plasma(norm(k_values))

        ax.scatter(x, y, color=colors, label=f'Time {time}', alpha=0.7, marker='o')

    ax.set_xlabel('$X_1$',fontsize=16)
    ax.set_ylabel('$X_2$',fontsize=16)

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array(all_k_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted rate',fontsize=16)

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{norm(x):.2f}'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file,bbox_inches='tight',transparent=True)
    plt.close()

# Compare observed snapshots, generated samples, and trajectories.
def plot_comparisions(
    df, generated, trajectories,
    palette = 'viridis',
    df_time_key='samples',
    x='x1', y='x2',
    groups=None,
    save=True, path='comparision.png',
    legend_loc = 'upper right'
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.viridis
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size':12,
        'axes.titlesize':10,
        'axes.labelsize':12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 300
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 300
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2


    axs = []
    for i, gs in enumerate(gspec):
        ax = plt.subplot(gs)


        n = 0.3

        ax.scatter(
                df[x], df[y],
                c=df[df_time_key],
                s=size,
                alpha=0.7 * n,
                marker='X',
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )

        for trajectory in np.transpose(trajectories, axes=(1,0,2)):
                plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black');

        states = sorted(df[df_time_key].unique())
        points = np.concatenate(generated, axis=0)
        n_gen = int(points.shape[0] / len(states))
        colors = [state for state in states for i in range(n_gen)]
        n = 1
        o = '.'
        ax.scatter(
                points[:, 0], points[:, 1],
                c='black',
                s=bg_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        ax.scatter(
                points[:, 0], points[:, 1],
                c='white',
                s=gap_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        pnts = ax.scatter(
                points[:, 0], points[:, 1],
                c=colors,
                s=size,
                alpha=0.7 * n,
                marker=o,
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )

        legend_elements = [
            Line2D(
                [0], [0], marker='o',
                color=cmap((i) / (len(states)-1)), label=f'T{state}',
                markerfacecolor=cmap((i) / (len(states)-1)), markersize=15,
            )
            for i, state in enumerate(states)
        ]

        leg = plt.legend(handles=legend_elements, loc='upper left', fontsize=16)
        ax.add_artist(leg)

        legend_elements = [
            Line2D(
                [0], [0], marker='X', color='w',
                label='Ground Truth', markerfacecolor=cmap(0), markersize=15, alpha=0.3
            ),
            Line2D([0], [0], marker='o', color='w', label='Predicted', markerfacecolor=cmap(.999), markersize=15),
            Line2D([0], [0], color='black', lw=2, label='Trajectory')

        ]
        leg = plt.legend(handles=legend_elements, loc=legend_loc, fontsize=16)
        ax.add_artist(leg)

        ax.set_xlabel("Gene $X_1$", fontsize=20)
        ax.set_ylabel("Gene $X_2$", fontsize=20)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.tick_params(which="both", **kwargs)

        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.patch.set_alpha(0)


        axs.append(ax)

    if save:

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches='tight')
        except ValueError:
            pass
    plt.close()
    return fig


def plot_branching_phase_portrait(
    df, trajectories,simulation_t,
    palette='viridis',
    df_time_key='samples',
    x='x1', y='x2',
    groups=None,
    save=True, path='branching_comparision.png',
    legend_loc = 'upper right'
):


    if groups is None:
        groups = sorted(df[df_time_key].unique())

    try:
        cmap = plt.get_cmap(palette)
    except:
        cmap = plt.cm.viridis

    sns.set_palette(palette)

    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })


    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.subplot(111)


    ax.scatter(
        df[x], df[y],
        c=df[df_time_key],
        s=100,
        alpha=0.2,
        marker='o',
        linewidths=0,
        cmap=cmap,
        label='_nolegend_'
    )


    if isinstance(trajectories, dict):

        all_times = [t for data in trajectories.values() for t in data['t']]
        if len(all_times) > 0:
            global_min_t = min(all_times)
        else:
            global_min_t = 0.0


        time_tolerance = 1e-3

        print(f"Plotting {len(trajectories)} trajectories...")

        for pid, data in trajectories.items():

            path_x = np.array(data['x'])
            times = data['t']


            if path_x.ndim > 1 and path_x.shape[1] >= 2:
                X_vals = path_x[:, 0]
                Y_vals = path_x[:, 1]


                ax.plot(X_vals, Y_vals, color='red', alpha=1, lw=1.5, zorder=2)


                start_t = times[0]
                if start_t > global_min_t + time_tolerance:

                    ax.scatter(X_vals[0], Y_vals[0],
                               marker='*', color='gold', edgecolor='firebrick',
                               s=120, zorder=4, label='_nolegend_')


                end_t = times[-1]
                if end_t < simulation_t - time_tolerance:

                    ax.scatter(X_vals[-1], Y_vals[-1],
                               marker='X', color='black',
                               s=80, zorder=4, label='_nolegend_')
            else:

                continue
    else:
        print("Error: trajectories format must be a dictionary (SDE mode=3 output).")


    gt_gradient = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(xx), markersize=10, alpha=0.5)
                   for xx in np.linspace(0, 1, 4)]

    legend_elements = [


        Line2D([0], [0], marker='o', color='w', label='Data Manifold',
               markerfacecolor='grey', markersize=10, alpha=0.3),


        Line2D([0], [0], color='red', lw=1.5, label='Particle Trajectory'),


        Line2D([0], [0], marker='*', color='w', label='Branching Event',
               markerfacecolor='gold', markeredgecolor='firebrick', markersize=15),


        Line2D([0], [0], marker='X', color='w', label='Death Event',
               markerfacecolor='black', markersize=10)
    ]

    ax.legend(handles=legend_elements, loc = legend_loc, frameon=True, fontsize=12)


    ax.set_xlabel(f"Dimension {x}", fontsize=16)
    ax.set_ylabel(f"Dimension {y}", fontsize=16)


    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    kwargs = dict(bottom=True, left=True)
    ax.tick_params(which="both", **kwargs)


    if save and path:
        try:
            fig.savefig(path, bbox_inches='tight')
            print(f"Branching plot saved to {path}")
        except Exception as e:
            print(f"Error saving: {e}")

    plt.close()
    return fig


def plot_branching_phase_portrait2(
    ax,
    df,
    trajectories,
    simulation_t,
    palette='viridis',
    df_time_key='samples',
    x='x1', y='x2',
    groups=None,
    ground_truth = False,
    color='red',
    legend_loc = 'upper right'
):


    if groups is None:
        groups = sorted(df[df_time_key].unique())

    try:
        cmap = plt.get_cmap(palette)
    except:
        cmap = plt.cm.viridis
        sns.set_palette(palette)


    if ground_truth:
        ax.scatter(
            df[x], df[y],
            c=df[df_time_key],
            s=100, alpha=0.2, marker='o', linewidths=0,
            cmap=cmap, label='_nolegend_'
        )


    if isinstance(trajectories, dict):

        all_times = [t for data in trajectories.values() for t in data['t']]
        global_min_t = min(all_times) if len(all_times) > 0 else 0.0

        time_tolerance = 1e-3
        print(f"Plotting {len(trajectories)} trajectories on existing ax...")

        for pid, data in trajectories.items():

            try:
                path_x = np.array(data['x'])
            except ValueError:
                clean_x_list = [np.array(item).flatten() for item in data['x']]
                path_x = np.stack(clean_x_list)

            times = data['t']


            if path_x.ndim > 1 and path_x.shape[1] >= 2:
                X_vals = path_x[:, 0]
                Y_vals = path_x[:, 1]


                ax.plot(X_vals, Y_vals, c=color, alpha=1, lw=1.5, zorder=2)


                start_t = times[0]
                if start_t > global_min_t + time_tolerance:
                    ax.scatter(X_vals[0], Y_vals[0],
                               marker='*', color='gold', edgecolor='firebrick',
                               s=120, zorder=4, label='_nolegend_')


                end_t = times[-1]


                if end_t < simulation_t - time_tolerance:
                    ax.scatter(X_vals[-1], Y_vals[-1],
                               marker='X', color='black',
                               s=80, zorder=4, label='_nolegend_')


                else:
                    ax.scatter(X_vals[-1], Y_vals[-1],
                               marker='o', color=color, edgecolor='white',
                               s=60, zorder=4, label='_nolegend_')

            else:
                continue
    else:
        print("Error: trajectories format must be a dictionary.")


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Data Manifold',
               markerfacecolor='grey', markersize=10, alpha=0.3),
        Line2D([0], [0], color='red', lw=1.5, label='Particle Trajectory'),
        Line2D([0], [0], marker='*', color='w', label='Branching Event',
               markerfacecolor='gold', markeredgecolor='firebrick', markersize=15),
        Line2D([0], [0], marker='X', color='w', label='Death Event',
               markerfacecolor='black', markersize=10),

        Line2D([0], [0], marker='o', color='w', label='Final Position (Alive)',
               markerfacecolor=color, markeredgecolor='white', markersize=8)
    ]

    ax.legend(handles=legend_elements, loc=legend_loc, frameon=True, fontsize=12)


    ax.set_xlabel(f"Dimension {x}", fontsize=16)
    ax.set_ylabel(f"Dimension {y}", fontsize=16)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    kwargs = dict(bottom=True, left=True)
    ax.tick_params(which="both", **kwargs)

    return ax

def compute_k_values(Xs, model):
    data_by_time = {}
    n = Xs[0].shape[1]

    for time in range(len(Xs)):
        subset = Xs[time]
        column_names = [f'x{i}' for i in range(1, n + 1)]
        tensors = torch.tensor(subset, dtype=torch.float32)

        with torch.no_grad():
            t = time*torch.ones((Xs[time].shape[0],1))
            _, _, k = model.forward(tensors, t)

        data_by_time[time] = {'data': subset, 'k_values': k.detach().cpu().numpy()}

    all_k_values = np.concatenate([content['k_values'] for content in data_by_time.values()])
    return all_k_values


# Provide a balanced Brownian bridge baseline.
class Brownian_Bridge(nn.Module):
    def __init__(self, dims, nu, activation = 'leakyrelu'):
        super(Brownian_Bridge, self).__init__()
        self.dims = dims
        self.nu = nu
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        blocks_v = []
        for i in range(len(dims)-1):
            blocks_v.append(nn.Linear(dims[i], dims[i+1]))
            blocks_v.append(self.activation)
        blocks_v.append(nn.Linear(dims[-1], dims[0] - 1))
        self.v_net = nn.Sequential(*blocks_v)

        blocks_s = []
        for i in range(len(dims)-1):
            blocks_s.append(nn.Linear(dims[i], dims[i+1]))
            blocks_s.append(self.activation)
        blocks_s.append(nn.Linear(dims[-1], 1))
        self.s_net = nn.Sequential(*blocks_s)


    def forward(self, x, t):


        v = self.v_net(torch.concatenate([x, t], dim=-1))
        s = self.s_net(torch.concatenate([x, t], dim=-1))
        return v, s

    def sample_comditional_flow(self, x0, x1, m0, m1, t):


        mean = t * x1 + (1-t) * x0
        sigma = torch.sqrt(t * (1-t)) * self.nu

        eps = torch.randn_like(mean)

        xts = mean + sigma*eps

        vts = (1-2*t)/(t*(1-t)+1e-8)*(xts-mean)+(x1-x0)

        mts = t * m1 + (1-t) * m0

        return xts, vts, 0, eps, mts

    def sample(self, x, t, N = 1000):
        if x is None:
            x = torch.randn(self.dims[0] - 1)
        if isinstance(t, float):
            t = torch.tensor([t])
        dt = t / N

        m = torch.ones_like(t)
        for i in range(N):
            ipt = torch.cat([x, i * t / N], dim=-1)
            v, s = self.forward(ipt)
            u = v - s
            x = x + u * dt + self.nu * torch.sqrt(dt) * torch.randn(x.shape)
            m += 0

        return x, m

# Provide an unbalanced bridge with a learned mass field.
class Unbalanced_Bridge(nn.Module):
    def __init__(self, dims, nu, activation = 'leakyrelu'):
        super(Unbalanced_Bridge, self).__init__()
        self.dims = dims
        self.nu = nu
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        blocks_v = []
        for i in range(len(dims)-1):
            blocks_v.append(nn.Linear(dims[i], dims[i+1]))
            blocks_v.append(self.activation)
        blocks_v.append(nn.Linear(dims[-1], dims[0] - 1))
        self.v_net = nn.Sequential(*blocks_v)


        blocks_k = []
        for i in range(len(dims)-1):
            blocks_k.append(nn.Linear(dims[i], dims[i+1]))
            blocks_k.append(self.activation)
        blocks_k.append(nn.Linear(dims[-1], 1))
        self.k_net = nn.Sequential(*blocks_k)


    def forward(self, x, t):


        v = self.v_net(torch.concatenate([x, t], dim=-1))
        k = self.k_net(torch.concatenate([x, t], dim=-1))
        return v, k

    def sample_comditional_flow(self, x0, x1, m0, m1, t, delta):


        diff = x1 - x0

        norm = torch.norm(diff, dim=1, keepdim=True)

        norm_vector = diff / (norm + 1e-9)


        tau = torch.tan(norm / (2 * delta))


        count = torch.sum(norm > (torch.pi * delta))

        if count.item() > 0:

            print('waring')


        scale = torch.sqrt(m0 * m1 / (1 + tau**2))


        omega = 2 * delta * tau * scale

        omega_vector = omega * norm_vector


        A = m1 + m0 - 2 * scale


        B = m0 - scale


        inv_sqrt_Am0_m_Bsq = 2*delta / (omega + 1e-9)


        xts = x0 + omega_vector * (inv_sqrt_Am0_m_Bsq * (torch.arctan((A*t - B)*inv_sqrt_Am0_m_Bsq) - torch.arctan(-B*inv_sqrt_Am0_m_Bsq)))


        mts = A*t**2 - 2*B*t + m0

        dm_dt = (2*A*t - 2*B)

        kts = dm_dt / mts


        vts = omega_vector / mts


        return xts, vts, kts, 0, mts


    def sample(self, x, t, N = 1000):
        if x is None:
            x = torch.randn(self.dims[0] - 1)
        if isinstance(t, float):
            t = torch.tensor([t])
        dt = t / N

        m = torch.ones_like(t)
        for i in range(N):
            ipt = torch.cat([x, i * t / N], dim=-1)
            v, k = self.forward(ipt)
            u = v
            x = x + u * dt
            m += k * dt
        return x, m


# Integrate deterministic trajectories from a learned velocity field.
class ODE:
    def __init__(self, model, nu, mode = 1, positive=True):
        self.drift = model.v_net
        self.rate = model.k_net
        self.nu = nu
        self.mode = mode
        self.positive = positive

    def trajectory(self,x,m,delta,N=100,T=1,I=None):
        xs = [x.detach().numpy().copy()]
        ms = [m.detach().numpy().copy()]

        if I:
            dt = (I[1]-I[0])/N
            ts = torch.linspace(I[0],I[1]-dt,N)
        else:
            dt = T/N
            ts = torch.linspace(0,T-dt,N)


        print('Continuous inference')
        for tt in ts:
            t = tt*torch.ones([x.size(0),1])
            v = self.drift(torch.concatenate([x,t],dim=-1))
            k = self.rate(torch.concatenate([x,t],dim=-1))

            x += (v)*dt
            m += k*dt
            if self.positive:
                m = torch.max(torch.zeros_like(m), m)

            xs.append(x.detach().numpy().copy())
            ms.append(m.detach().numpy().copy())
        return np.array(xs), np.array(ms).squeeze()

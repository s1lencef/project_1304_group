"""
Сравнение BCDA (Broad Critic Deep Actor, arXiv:2411.15806)
и PPO в среде Pendulum-v1.


КЛЮЧЕВОЕ ОТЛИЧИЕ АЛГОРИТМОВ
  PPO   — критик это DNN, обучается через backprop (gradient descent),
          on-policy, политика стохастическая (Gaussian).
  BCDA  — критик это Broad Learning System (BLS): входные проекции
          фиксированы, выходные веса считаются АНАЛИТИЧЕСКИ через
          гребневую регрессию. Off-policy, актор детерминированный (DNN).
"""
import random, collections
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SEEDS = [7, 17, 42]
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class BLS:
    def __init__(self, sd, ad, nz=40, nh=80, lam=1e-4):
        self.sd=sd; self.nz=nz; self.nh=nh; self.lam=lam
        rng=np.random.default_rng(0); idim=sd+ad
        self.Wz=rng.standard_normal((idim,nz))*.1; self.bz=rng.standard_normal((1,nz))*.1
        self.Wh=rng.standard_normal((nz,nh))*.1;   self.bh=rng.standard_normal((1,nh))*.1
        self.W_out=np.zeros((nz+nh,1))
    def _phi(self,s,a):
        x=np.concatenate([s,a],1)
        Z=np.tanh(x@self.Wz+self.bz); H=np.tanh(Z@self.Wh+self.bh)
        return np.concatenate([Z,H],1)
    def predict(self,s,a): return self._phi(s,a)@self.W_out
    def update(self,s,a,y):
        A=self._phi(s,a); I=np.eye(self.nz+self.nh)
        self.W_out=np.linalg.solve(A.T@A+self.lam*I,A.T@y)
    def grad_a(self,s,a):
        x=np.concatenate([s,a],1)
        Z=np.tanh(x@self.Wz+self.bz); H=np.tanh(Z@self.Wh+self.bh)
        dZ=(1-Z**2)[:,:,None]*self.Wz.T[None]
        dH=np.einsum('nik,nkj->nij',(1-H**2)[:,:,None]*self.Wh.T[None],dZ)
        dQ=np.einsum('nki,k->ni',np.concatenate([dZ,dH],1),self.W_out.flatten())
        return dQ[:,self.sd:]

def make_actor(sd,ad,scale,h=256):
    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale=scale
            self.net=nn.Sequential(nn.Linear(sd,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,ad),nn.Tanh())
        def forward(self,x): return self.net(x)*self.scale
    return A()

def run_bcda(env_name, n_ep, seed,
             nz=60, nh=120, lam=1e-4,
             actor_lr=1e-3, gamma=0.99, tau=0.005,
             bs=128, buf_cap=100_000, noise_std=0.2,
             critic_freq=4, actor_freq=2):
    set_seed(seed)
    env=gym.make(env_name); env.action_space.seed(seed)
    sd=env.observation_space.shape[0]; ad=env.action_space.shape[0]
    scale=float(env.action_space.high[0])

    critic=BLS(sd,ad,nz,nh,lam); tc=BLS(sd,ad,nz,nh,lam)
    tc.W_out=critic.W_out.copy()
    actor=make_actor(sd,ad,scale); ta=make_actor(sd,ad,scale)
    ta.load_state_dict(actor.state_dict())
    opt=optim.Adam(actor.parameters(),actor_lr)

    buf=collections.deque(maxlen=buf_cap)
    history=[]; step=0

    for ep in range(n_ep):
        s,_=env.reset(seed=seed+ep); ep_r=0.0
        for _ in range(999):
            with torch.no_grad():
                a=actor(torch.FloatTensor(s)).numpy()
            a+=np.random.normal(0,noise_std*scale,size=a.shape)
            a=np.clip(a,-scale,scale)
            ns,r,t,tr,_=env.step(a); done=t or tr
            buf.append((s.astype('f'),a.astype('f'),r,ns.astype('f'),float(done)))
            ep_r+=r; s=ns; step+=1

            if len(buf)>=bs:
                # Один сэмпл на оба обновления
                batch=random.sample(buf,bs)
                sb,ab,rb,nsb,db=[np.array(x,'f') for x in zip(*batch)]

                if step%critic_freq==0:
                    with torch.no_grad(): na=ta(torch.FloatTensor(nsb)).numpy()
                    tq=tc.predict(nsb,na)*(1-db.reshape(-1,1))
                    critic.update(sb,ab,rb.reshape(-1,1)+gamma*tq)
                    tc.W_out=tau*critic.W_out+(1-tau)*tc.W_out

                if step%actor_freq==0:
                    opt.zero_grad()
                    pred=actor(torch.FloatTensor(sb))
                    dq=critic.grad_a(sb,pred.detach().numpy())
                    loss=-(pred*torch.FloatTensor(dq)).mean()
                    loss.backward(); opt.step()
                    for p,tp in zip(actor.parameters(),ta.parameters()):
                        tp.data.copy_(tau*p+(1-tau)*tp.data)
            if done: break
        history.append(ep_r)
    env.close(); return history

class PolicyNet(nn.Module):
    def __init__(self,od,ad,h=64):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(od,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,ad),nn.Tanh())
        self.log_std=nn.Parameter(torch.zeros(ad))
    def forward(self,x):
        m=self.trunk(x); return Normal(m,torch.exp(self.log_std).expand_as(m))

class ValueNet(nn.Module):
    def __init__(self,od,h=64):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(od,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(self,x): return self.trunk(x).squeeze(-1)

def run_ppo(env_name, n_ep, seed,
            h=64, lrp=3e-4, lrv=1e-3, gamma=0.99, lam=0.95,
            clip=0.2, steps=2048, epochs=10, mb=64, gnorm=0.5):
    set_seed(seed)
    env=gym.make(env_name)
    od=env.observation_space.shape[0]; ad=env.action_space.shape[0]
    policy=PolicyNet(od,ad,h); critic=ValueNet(od,h)
    op=optim.Adam(policy.parameters(),lrp); ov=optim.Adam(critic.parameters(),lrv)

    # Rollout buffer
    buf_o=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]
    history=[]; obs,_=env.reset(seed=seed); obs=obs.astype('f')
    ep_r=0.0; cnt=0

    while cnt<n_ep:
        for _ in range(steps):
            t=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
            dist=policy(t); a=dist.sample()
            lp=dist.log_prob(a).sum(-1).item(); v=critic(t).item()
            ns,r,ter,tru,_=env.step(a.squeeze(0).numpy()); done=ter or tru
            ns=ns.astype('f')
            buf_o.append(obs.copy()); buf_a.append(a.squeeze(0).numpy())
            buf_r.append(r); buf_d.append(done); buf_lp.append(lp); buf_v.append(v)
            ep_r+=r; obs=ns
            if done:
                history.append(ep_r); cnt+=1; ep_r=0.0
                obs,_=env.reset(); obs=obs.astype('f')
                if cnt>=n_ep: break

        T=len(buf_r); advs=np.zeros(T,'f')
        with torch.no_grad():
            lv=critic(torch.tensor(obs,dtype=torch.float32).unsqueeze(0)).item()
        g=0.0
        for i in reversed(range(T)):
            nv=lv if i==T-1 else buf_v[i+1]
            m=1.-float(buf_d[i])
            d=buf_r[i]+gamma*nv*m-buf_v[i]; g=d+gamma*lam*m*g; advs[i]=g
        rets=advs+np.array(buf_v,'f')
        advs=(advs-advs.mean())/(advs.std()+1e-8)
        pass  # replaced
        obs_t=torch.tensor(np.array(buf_o),dtype=torch.float32)
        acts_t=torch.tensor(np.array(buf_a),dtype=torch.float32)
        olp_t=torch.tensor(np.array(buf_lp),dtype=torch.float32)
        adv_t=torch.tensor(advs,dtype=torch.float32)
        ret_t=torch.tensor(rets,dtype=torch.float32)
        N=len(buf_o)
        for _ in range(epochs):
            perm=torch.randperm(N)
            for st in range(0,N,mb):
                idx=perm[st:st+mb]
                dist2=policy(obs_t[idx]); nlp=dist2.log_prob(acts_t[idx]).sum(-1)
                ratio=torch.exp(nlp-olp_t[idx])
                lpi=-torch.min(ratio*adv_t[idx],torch.clamp(ratio,1-clip,1+clip)*adv_t[idx]).mean()
                op.zero_grad(); lpi.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),gnorm); op.step()
                lv2=nn.functional.mse_loss(critic(obs_t[idx]),ret_t[idx])
                ov.zero_grad(); lv2.backward()
                nn.utils.clip_grad_norm_(critic.parameters(),gnorm); ov.step()
        buf_o=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]

    env.close(); return history

def smooth(arr, w=20):
    out=np.full(len(arr),np.nan)
    for i in range(w-1,len(arr)): out[i]=np.mean(arr[i-w+1:i+1])
    return out

def save_plots(bcda_all, ppo_all, n_ep, show=True):
    """
    Сохраняет 3 графика в текущую папку (в Colab — /content/).
    show=True дополнительно отрисует их прямо в ячейке ноутбука.
    """
    OUT = "."
    ba = np.array(bcda_all); pa = np.array(ppo_all)
    eps = np.arange(1, n_ep + 1)
    C = ['#2196F3', '#FF5722']
    paths = []

    # ── График 1: основной с полосами ± σ ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for arr, c, label in [
        (ba, C[0], 'BCDA (BLS-критик, arXiv:2411.15806)'),
        (pa, C[1], 'PPO  (DNN-критик, практ. работа №2)')]:
        m = arr.mean(0); s = arr.std(0)
        ax.fill_between(eps, m - s, m + s, alpha=0.15, color=c)
        ax.plot(eps, m, alpha=0.2, color=c, linewidth=0.7)
        ax.plot(eps, smooth(m, 25), label=label, color=c, linewidth=2.2)
    ax.axhline(-200, color='green', ls='--', lw=1, alpha=0.7, label='Ориентир (-200)')
    ax.set_title(f'BCDA vs PPO — Pendulum-v1\n'
                 f'(среднее по {len(SEEDS)} сидам {SEEDS}, полоса ± σ)', fontsize=12)
    ax.set_xlabel('Эпизод'); ax.set_ylabel('Суммарная награда')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = f'{OUT}/comparison_bcda_ppo.png'; plt.savefig(p, dpi=150); paths.append(p)
    if show: plt.show()
    plt.close()

    # ── График 2: разбивка по сидам ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ls = ['-', '--', ':']
    for ax, (arrs, title, c) in zip(axes, [
        (bcda_all, 'BCDA (BLS-критик)', C[0]),
        (ppo_all,  'PPO  (DNN-критик)', C[1])]):
        for i, (seed, h) in enumerate(zip(SEEDS, arrs)):
            ax.plot(smooth(h, 20), ls=ls[i], color=c, lw=1.6, label=f'seed={seed}')
        ax.axhline(-200, color='green', ls='--', lw=1, alpha=0.6)
        ax.set_title(title); ax.set_xlabel('Эпизод')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_ylabel('Награда (сглажено)')
    plt.suptitle('Разбивка по сидам — Pendulum-v1', fontsize=12)
    plt.tight_layout()
    p = f'{OUT}/comparison_by_seed.png'; plt.savefig(p, dpi=150); paths.append(p)
    if show: plt.show()
    plt.close()

    # ── График 3: box plots ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [('Avg last 50 эп.', ba[:, -50:].mean(1), pa[:, -50:].mean(1)),
               ('Max награда',      ba.max(1),           pa.max(1)),
               ('Avg first 50 эп.', ba[:, :50].mean(1),  pa[:, :50].mean(1))]
    for ax, (title, bv, pv) in zip(axes, metrics):
        # tick_labels вместо устаревшего labels (удалён в matplotlib 3.11)
        bp = ax.boxplot([bv, pv], tick_labels=['BCDA', 'PPO'], patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#BBDEFB')
        bp['boxes'][1].set_facecolor('#FFCCBC')
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    plt.suptitle('Распределение метрик по сидам', fontsize=11)
    plt.tight_layout()
    p = f'{OUT}/comparison_metrics.png'; plt.savefig(p, dpi=150); paths.append(p)
    if show: plt.show()
    plt.close()

    print('  Все графики сохранены:', ', '.join(paths))

ENV='Pendulum-v1'; N_EP=250; MAX_STEP=200

if __name__=='__main__':
    print('='*60)
    print(f'BCDA vs PPO | {ENV} | {N_EP} эп. | сиды {SEEDS}')
    print('='*60)
    bcda_all=[]; ppo_all=[]

    for seed in SEEDS:
        print(f'\n── Сид {seed} ─────────────────────────')
        print('  [BCDA]...'); hb=run_bcda(ENV,N_EP,seed)
        bcda_all.append(hb)
        print(f'  avg_last50={np.mean(hb[-50:]):.2f}  max={max(hb):.2f}')
        print('  [PPO]...'); hp=run_ppo(ENV,N_EP,seed)
        ppo_all.append(hp)
        print(f'  avg_last50={np.mean(hp[-50:]):.2f}  max={max(hp):.2f}')

    ba=np.array(bcda_all); pa=np.array(ppo_all)
    print('\n'+'─'*56)
    print(f"{'Метрика':<38} {'BCDA':>8} {'PPO':>8}")
    print('─'*56)
    for lbl,bv,pv in [
        ('Avg last 50 (mean по сидам)', ba[:,-50:].mean(), pa[:,-50:].mean()),
        ('Avg last 50 (σ   по сидам)', ba[:,-50:].mean(1).std(), pa[:,-50:].mean(1).std()),
        ('Max (mean по сидам)',         ba.max(1).mean(), pa.max(1).mean()),
        ('Avg first 50',                ba[:,:50].mean(), pa[:,:50].mean()),
    ]:
        print(f"  {lbl:<36} {bv:>8.2f} {pv:>8.2f}")
    print('─'*56)
    print('\nСохранение графиков...')
    save_plots(bcda_all,ppo_all,N_EP)
    print('\nГотово!')

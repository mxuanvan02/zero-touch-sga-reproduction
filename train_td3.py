"""Train the TD3 adaptive sensor-activation policy (Algorithm 1).

The agent learns an adaptive sensing duty cycle + semantic fidelity. Its
learned behaviour produces the "Proposed" row of the paper's Table III
(energy breakdown), and the headline energy saving is computed as
Proposed vs CENT (the closest competitive baseline, = 18.2% in the paper).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import config as C
from env import SensorActivationEnv
from td3 import TD3, ReplayBuffer
from energy_model import baselines, headline_savings


def train_td3(episodes=C.TD3_EPISODES, start_random=15, seed=C.SEED, verbose=True):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    env = SensorActivationEnv(seed=seed)
    agent = TD3(env.state_dim, env.action_dim)
    buf = ReplayBuffer(env.state_dim, env.action_dim)

    ep_returns, ep_energy, ep_duty = [], [], []
    for ep in range(episodes):
        s = env.reset(); done = False
        ep_ret, ep_e, duties = 0.0, 0.0, []
        expl = max(0.05, 0.3 * (1 - ep / episodes))
        while not done:
            a = (np.random.uniform(-1, 1, env.action_dim) if ep < start_random
                 else agent.act(s, noise=expl))
            s2, r, done, info = env.step(a)
            buf.add(s, a, r, s2, float(done)); s = s2
            ep_ret += r; ep_e += info["energy"]; duties.append(info["duty"])
            if len(buf) > 256:
                agent.train(buf)
        ep_returns.append(ep_ret); ep_energy.append(ep_e)
        ep_duty.append(float(np.mean(duties)))
        if verbose and (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d} | return {ep_ret:7.2f} | duty "
                  f"{ep_duty[-1]:.2f} | energy {ep_e:5.2f} Wh")

    # greedy evaluation: learned sensing duty, gamma, and deficit
    duties, gammas, defs = [], [], []
    for _ in range(25):
        s = env.reset(); done = False; ds, gs = [], []
        while not done:
            a = agent.act(s, noise=0.0)
            s, r, done, info = env.step(a); ds.append(info["duty"]); gs.append(info["gamma"])
        duties.append(np.mean(ds)); gammas.append(np.mean(gs))
        defs.append(float(np.clip(C.MIN_RECORDS - env.records, 0, None).sum()))
    proposed_duty = float(np.mean(duties))
    mean_gamma = float(np.mean(gammas))
    mean_deficit = float(np.mean(defs))

    # Map the learned duty through the calibrated Table III model.
    rows = baselines(proposed_duty)
    savings = headline_savings(rows)            # Proposed vs CENT (paper 18.2%)
    return {
        "proposed_duty": proposed_duty,
        "mean_gamma": mean_gamma,
        "mean_deficit": mean_deficit,
        "table3": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in rows.items()},
        "proposed_total": rows["Proposed"]["total"],
        "cent_total": rows["CENT"]["total"],
        "savings": float(savings),
        "energy_curve": ep_energy,
        "duty_curve": ep_duty,
        "returns": ep_returns,
    }


if __name__ == "__main__":
    r = train_td3()
    print("\n  Approach    Fly  Hover  Sense  Comm   Total")
    fh = C.E_FLY_HOVER
    for name, row in r["table3"].items():
        print(f"  {name:<9} {8.2:5.1f}{6.1:6.1f}{row['sense']:7.2f}{row['comm']:6.2f}{row['total']:7.2f}")
    print(f"\n  TD3 learned sensing duty: {r['proposed_duty']:.2f}, gamma: {r['mean_gamma']:.2f}")
    print(f"  Deficit at eval: {r['mean_deficit']:.1f} (0 = mission complete)")
    print(f"  Headline savings (Proposed vs CENT): {100*r['savings']:.1f}%  (paper 18.2%)")

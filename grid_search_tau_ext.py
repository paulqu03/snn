# ------------------------------------------------------------------
# Imports & helpers
# ------------------------------------------------------------------
import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Helper -------------------------------------------------

def create_directory(path: str | Path):
    """Create directory *path* (incl. parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def pop_rate(spike_monitor: SpikeMonitor, t0, t1, n_neurons: int) -> float:
    """Return mean firing rate (Hz) in [t0, t1)."""
    count = np.sum((spike_monitor.t >= t0) & (spike_monitor.t < t1))
    return count / ((t1 - t0) / second) 


# ------------------------------------------------------------------
# Experiment setup (unchanged defaults)
# ------------------------------------------------------------------
base_path = Path("./experiments/decision_simulation/grid_search_tau/v_post_E")
create_directory(base_path)

# Network sizes
N_E1, N_E2, N_I = 100, 100, 20

# Time constants (E populations fixed, τ_I is the grid parameter)
tau_E1, tau_E2 = 20*ms, 20*ms
tau_I = 10*ms

# Other cell parameters
v_threshold_E1 = 15*mV
v_threshold_E2 = 15*mV
v_threshold_I  = 15*mV

v_reset_E1 = 5*mV
v_reset_E2 = 5*mV
v_reset_I  = 5*mV

refractory_time_E1 = 0*ms
refractory_time_E2 = 0*ms
refractory_time_I  = 0*ms

# Stimulus / runtime
stim_on, stim_off = 500*ms, 1500*ms
runtime = 2000*ms

mu0, mu1 = 100*Hz, 85*Hz        # E1‑Vorteil
sigma = 4*Hz
stim_interval = 50*ms

# Background Poisson
bg_rate = 55*Hz
v_post_poisson = 1.4*mV

# Synaptic weights (baseline)
v_post_stim = 3.8*mV
v_post_E1, v_post_E2 = 2.3*mV, 2.3*mV  # **grid parameter v_post_E1 / v_post_E2**
v_post_I  = 2.0*mV
v_post_E1_self, v_post_E2_self = 0.5*mV, 0.5*mV
v_post_I_self = 0.01*mV

# Connection probabilities
p_P_E1 = p_P_E2 = 0.05
p_E1_E1 = p_E2_E2 = 0.35
p_I_I   = 0.05
p_E1_I = p_E2_I = 0.15
p_I_E1 = p_I_E2 = 0.13

# ------------------------------------------------------------------
# Network template – we build once, then clone/restore
# ------------------------------------------------------------------


# --- neuron groups
E1 = NeuronGroup(N_E1, 'dv/dt = -v / tau_E1 : volt (unless refractory)',
                threshold='v >= v_threshold_E1', reset='v = v_reset_E1',
                method='euler', refractory=refractory_time_E1, name='E1')
E2 = NeuronGroup(N_E2, 'dv/dt = -v / tau_E2 : volt (unless refractory)',
                threshold='v >= v_threshold_E2', reset='v = v_reset_E2',
                method='euler', refractory=refractory_time_E2, name='E2')
I  = NeuronGroup(N_I,  'dv/dt = -v / tau_I : volt (unless refractory)',
                threshold='v >= v_threshold_I', reset='v = v_reset_I',
                method='euler', refractory=refractory_time_I, name='I')

E1.v = 'v_threshold_E1-5*mV'
E2.v = 'v_threshold_E2-5*mV'
I.v = 'v_threshold_I-5*mV'

# --- background Poisson
P_E1 = PoissonGroup(100, rates=bg_rate, name='P_E1')
P_E2 = PoissonGroup(100, rates=bg_rate, name='P_E2')

S_P_E1 = Synapses(P_E1, E1, on_pre='v += v_post_poisson', name='S_P_E1')
S_P_E2 = Synapses(P_E2, E2, on_pre='v += v_post_poisson', name='S_P_E2')
S_P_E1.connect(p=p_P_E1)
S_P_E2.connect(p=p_P_E2)

# --- stimulus Poisson groups (rates updated every 50 ms)
StE1 = PoissonGroup(20, rates=0*Hz, name='StE1')
StE2 = PoissonGroup(20, rates=0*Hz, name='StE2')

StE1.run_regularly(
    "rates = int(t > stim_on and t < stim_off) * "
    "(mu0 + sigma*randn())",
    dt=stim_interval
)
StE2.run_regularly(
    "rates = int(t > stim_on and t < stim_off) * "
    "(mu1 + sigma*randn())",
    dt=stim_interval
)

S_StE1 = Synapses(StE1, E1, on_pre='v += v_post_stim', delay=0.5*ms, name='S_StE1')
S_StE2 = Synapses(StE2, E2, on_pre='v += v_post_stim', delay=0.5*ms, name='S_StE2')
S_StE1.connect(p=p_P_E1)
S_StE2.connect(p=p_P_E2)

# --- recurrent & cross connections (weights will be modified on the fly)
S_E1_E1 = Synapses(E1, E1, on_pre='v += v_post_E1_self', delay=0.5*ms, name='S_E1_E1')
S_E2_E2 = Synapses(E2, E2, on_pre='v += v_post_E2_self', delay=0.5*ms, name='S_E2_E2')
S_I_I   = Synapses(I, I,   on_pre='v -= v_post_I_self', delay=0.5*ms, name='S_I_I')

S_E1_E1.connect(condition='i != j', p=p_E1_E1)
S_E2_E2.connect(condition='i != j', p=p_E2_E2)
S_I_I.connect( condition='i != j', p=p_I_I)

S_E1_I = Synapses(E1, I, on_pre='v += v_post_E1', delay=0.5*ms, name='S_E1_I')
S_E2_I = Synapses(E2, I, on_pre='v += v_post_E2', delay=0.5*ms, name='S_E2_I')
S_E1_I.connect(p=p_E1_I)
S_E2_I.connect(p=p_E2_I)

S_I_E1 = Synapses(I, E1, on_pre='v -= v_post_I', delay=0.5*ms, name='S_I_E1')
S_I_E2 = Synapses(I, E2, on_pre='v -= v_post_I', delay=0.5*ms, name='S_I_E2')
S_I_E1.connect(p=p_I_E1)
S_I_E2.connect(p=p_I_E2)

# --- monitors
PR_E1, PR_E2, PR_I = (PopulationRateMonitor(grp, name=f'PR_{grp.name}')
                        for grp in (E1, E2, I))
SM_E1, SM_E2, SM_I = (SpikeMonitor(grp, name=f'SM_{grp.name}')
                        for grp in (E1, E2, I))
StM_E1 = StateMonitor(StE1, 'rates', record=0, dt=1*ms)
StM_E2 = StateMonitor(StE2, 'rates', record=0, dt=1*ms)

# --- collect into Net
net = Network()
net.add(E1, E2, I, P_E1, P_E2, 
        S_P_E1, S_P_E2, StE1, StE2, S_StE1, S_StE2, 
        S_E1_E1, S_E2_E2, S_E1_I, S_E2_I, S_I_I, S_I_E1, S_I_E2, 
        PR_E1, PR_E2, PR_I, 
        SM_E1, SM_E2, SM_I, StM_E1, StM_E2)
net.store()

# ------------------------------------------------------------------
# Set Parameters
# ------------------------------------------------------------------
TAU_I_VALUES  = np.arange(1, 11) * ms           # 1 .. 10 ms
GRID_VALUES   = np.arange(0.5, 7.51, 0.5)*mV  # 0.05 .. 0.25 
N_RUNS        = 20
BIN_SIZE      = 15*ms                           # smoothing for plots

n_tau = len(TAU_I_VALUES)
n_val = len(GRID_VALUES)

# Containers for heatmap data
accuracy_map   = np.zeros((n_tau, n_val))
clarity_map    = np.zeros((n_tau, n_val))
eff_inhib_map  = np.zeros((n_tau, n_val))
win_clarity_map = np.zeros((n_tau, n_val))
lose_clarity_map= np.zeros((n_tau, n_val))
sensitivity_map= np.zeros((n_tau, n_val))
over_sens_map  = np.zeros((n_tau, n_val))

# ------------------------------------------------------------------
# Main loops: τ_I × v_post_I
# ------------------------------------------------------------------
for tau_idx, tau_I in enumerate(TAU_I_VALUES):

    # Adjust I‑membrane τ (done by re‑creating I with new tau every outer loop)
    net.restore()
    # replace existing I group with new tau (Brian2 doesn't allow param change)
    net.remove(I, S_I_I, S_I_E1, S_I_E2, PR_I, SM_I)

    eqs_I = '''
    dv/dt = -v / tau_I : volt (unless refractory)
    '''

    I = NeuronGroup(N_I, eqs_I, threshold='v >= v_threshold_I', reset='v = v_reset_I', method='euler', refractory=refractory_time_I)
    I.v = 'v_threshold_I-5*mV'

    S_I_I = Synapses(I, I, on_pre='v -= v_post_I_self', delay=0.5*ms, name="S_I_I")
    S_I_I.connect(condition='i != j', p=p_I_I)

    S_I_E2 = Synapses(I, E2, on_pre='v -= v_post_I', delay=0.5*ms, name="S_I_E2")
    S_I_E1 = Synapses(I, E1, on_pre='v -= v_post_I', delay=0.5*ms, name="S_I_E1")
    S_I_E1.connect(p=p_I_E1)
    S_I_E2.connect(p=p_I_E2)

    PR_I = PopulationRateMonitor(I, name='PR_I')
    SM_I = SpikeMonitor(I, name='SM_I')

    net.add(I, S_I_I, S_I_E1, S_I_E2, PR_I, SM_I)

    # Update object references for inner loop convenience
    net.store()
    
    # --------------------------------------------------------------
    for val_idx, v_val in enumerate(GRID_VALUES):
        # Adjust E→I weights (both populations)
        net.restore()        # base state incl. new I
        # reconnect weight synapses E1→I / E2→I with fresh v_val
        net.remove(S_E1_I, S_E2_I)

        S_E1_I = Synapses(E1, I, on_pre='v += v_val', delay=0.5*ms, name='S_E1_I')
        S_E2_I = Synapses(E2, I, on_pre='v += v_val', delay=0.5*ms, name='S_E2_I')
        S_E1_I.connect(p=p_E1_I)
        S_E2_I.connect(p=p_E2_I)

        net.add(S_E1_I, S_E2_I)
        net.store()

        # --- buffers for metrics
        true_pos, comp_cnt, predec_cnt = 0, 0, 0
        delta_abs_list, i_eff_list = [], []
        win_act_list, lose_act_list = [], []

        # ----------------------------------------------------------
        for run_idx in range(N_RUNS):
            net.restore()
            net.run(runtime)

            # === helper lambda for rate in interval
            rate = lambda sm, a, b, N: pop_rate(sm, a, b, N)

            # Stimulus & baseline activities
            A_E1 = rate(SM_E1, stim_on, stim_off, N_E1)
            A_E2 = rate(SM_E2, stim_on, stim_off, N_E2)
            B_E1 = (
                rate(SM_E1, 0*ms, stim_on, N_E1) * (stim_on/runtime) +
                rate(SM_E1, stim_off, runtime, N_E1) * ((runtime - stim_off)/runtime)
            ) / ((stim_on/runtime) + ((runtime - stim_off)/runtime))
            B_E2 = (
                rate(SM_E2, 0*ms, stim_on, N_E2) * (stim_on/runtime) +
                rate(SM_E2, stim_off, runtime, N_E2) * ((runtime - stim_off)/runtime)
            ) / ((stim_on/runtime) + ((runtime - stim_off)/runtime))

            delta = A_E1 - A_E2
            delta_abs_list.append(abs(delta))

            # --- Accuracy
            if delta > 30:    # True Positive (E1 sollte gewinnen)
                true_pos += 1

            # --- Effective inhibition
            if delta >= 0:   # E2 lost
                i_eff_list.append(A_E2 - B_E2)
                win_act_list.append(A_E1)
                lose_act_list.append(A_E2)
            else:           # E1 lost
                i_eff_list.append(A_E1 - B_E1)
                win_act_list.append(A_E2)
                lose_act_list.append(A_E1)

            # --- Sensitivity (competition during stimulus)
            winners = []
            for step in range(10):
                t0 = stim_on + step*100*ms
                t1 = t0 + 100*ms
                a1 = rate(SM_E1, t0, t1, N_E1)
                a2 = rate(SM_E2, t0, t1, N_E2)
                winners.append(0 if a1 >= a2 else 1)  # 0:E1, 1:E2
            if len(set(winners)) > 1:
                comp_cnt += 1

            # --- Overly Sensitivity (decision before stim)
            pre_a1 = rate(SM_E1, 0*ms, stim_on, N_E1)
            pre_a2 = rate(SM_E2, 0*ms, stim_on, N_E2)
            if abs(pre_a1 - pre_a2) > 1000:
                predec_cnt += 1

            # --- save plots as before --------------------------------
            sim_folder = base_path / f"tau_I_{int(tau_I/ms)}ms" / f"v_post_E_{float(v_val/mV)}mV" / f"simulation_{run_idx+1}"
            create_directory(sim_folder)

            # Spike‑rate plot (smoothed)
            fig1 = plt.figure(figsize=(16, 9))
            ax1 = fig1.add_subplot(111)
            pr_e1 = PR_E1.smooth_rate(window='flat', width=BIN_SIZE)/Hz
            pr_e2 = PR_E2.smooth_rate(window='flat', width=BIN_SIZE)/Hz
            ax1.plot(PR_E1.t/ms, pr_e1, label='E1')
            ax1.plot(PR_E2.t/ms, pr_e2, label='E2')
            ax1.set(xlabel='Time (ms)', ylabel='Firing rate (Hz)',
                    title=f'Spike rates (bin {BIN_SIZE/ms} ms)')
            ax1.legend(); fig1.tight_layout()
            fig1.savefig(sim_folder / 'spike_activity.png'); plt.close(fig1)

            # Spike raster
            fig2 = plt.figure(figsize=(16, 9))
            ax2 = fig2.add_subplot(111)
            ax2.scatter(SM_E1.t/ms, SM_E1.i, s=2, label='E1', color='steelblue')
            ax2.scatter(SM_E2.t/ms, SM_E2.i+N_E1, s=2, label='E2', color='orange')
            ax2.scatter(SM_I.t/ms, SM_I.i+N_E1+N_E2, s=2, label='I', color='green')
            ax2.set(xlabel='Time (ms)', ylabel='Neuron index', title='Spike raster', xlim=(0, runtime/ms))
            ax2.legend(loc='upper right'); fig2.tight_layout()
            fig2.savefig(sim_folder / 'spike_raster_plot.png'); plt.close(fig2)

            # Stimulus plot
            fig3 = plt.figure(figsize=(16, 9))
            ax3 = fig3.add_subplot(111)
            ax3.plot(StM_E1.t/ms, StM_E1.rates[0]/Hz, color='steelblue', label='Input E1')
            ax3.plot(StM_E2.t/ms, StM_E2.rates[0]/Hz, color='orange', label='Input E2')
            ax3.set(xlabel='Time (ms)', ylabel='Rate (Hz)'); ax3.legend(); fig3.tight_layout()
            fig3.savefig(sim_folder / 'stimulus.png'); plt.close(fig3)

        # ----------------------------------------------------------
        # Aggregate over 20 simulations
        accuracy_map[tau_idx, val_idx]    = true_pos / N_RUNS * 100
        clarity_map[tau_idx, val_idx]     = np.mean(delta_abs_list)
        eff_inhib_map[tau_idx, val_idx]   = np.mean(i_eff_list)
        win_clarity_map[tau_idx, val_idx] = np.mean(win_act_list)
        lose_clarity_map[tau_idx, val_idx]= np.mean(lose_act_list)
        sensitivity_map[tau_idx, val_idx] = comp_cnt / N_RUNS * 100
        over_sens_map[tau_idx, val_idx]   = predec_cnt / N_RUNS * 100

# ------------------------------------------------------------------
# Heatmap plotting util
# ------------------------------------------------------------------

def save_heatmap(matrix: np.ndarray, title: str, cmap: str, fname: str, percent=False):
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(matrix, origin='lower', aspect='auto', cmap=cmap,
                   extent=(GRID_VALUES[0]/mV, GRID_VALUES[-1]/mV,
                           TAU_I_VALUES[0]/ms, TAU_I_VALUES[-1]/ms))
    ax.set_xlabel('v_post_E (mV)')
    ax.set_ylabel('tau_I (ms)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    if percent:
        cbar.set_label('%')
    else:
        cbar.set_label('value')
    fig.tight_layout()
    fig.savefig(heatmap_dir / fname)
    plt.close(fig)

# ------------------------------------------------------------------
# Save all heatmaps
# ------------------------------------------------------------------
heatmap_dir = base_path / 'heatmaps'
create_directory(heatmap_dir)

save_heatmap(accuracy_map,   'Accuracy Heatmap',             'viridis', 'heatmap_accuracy.png',   percent=True)
save_heatmap(clarity_map,    'Clarity Heatmap',              'plasma',   'heatmap_clarity.png')
save_heatmap(eff_inhib_map,  'Effective Inhibition Heatmap', 'plasma',  'heatmap_eff_inhib.png')
save_heatmap(win_clarity_map,'Winner Clarity Heatmap',       'inferno', 'heatmap_win_clarity.png')
save_heatmap(lose_clarity_map,'Loser Clarity Heatmap',       'inferno', 'heatmap_lose_clarity.png')
save_heatmap(sensitivity_map,'Sensitivity Heatmap',          'viridis','heatmap_sensitivity.png',percent=True)
save_heatmap(over_sens_map,  'Over‑Sensitivity Heatmap',     'viridis','heatmap_over_sens.png',  percent=True)

print("✅ Grid‑search abgeschlossen – Heatmaps liegen unter:", heatmap_dir)

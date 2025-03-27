from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import random

# Netzwerkspezifikationen
N_E = 100  # Anzahl exzitatorischer Neuronen
N_I = 25   # Anzahl inhibitorischer Neuronen
v_threshold = 15 * mV
V_reset = 0 * mV
tau = 20 * ms
refractory_time = 5 * ms
duration = 1000 * ms

# Neuronale Gleichungen
eqs = '''
dv/dt = -v / tau : volt (unless refractory)
'''

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

# Erstellen von exzitatorischen und inhibitorischen Neuronengruppen
E = NeuronGroup(N_E, eqs, threshold='v > v_threshold', reset='v = V_reset', 
                                 method='exact', refractory=refractory_time)
I = NeuronGroup(N_I, eqs, threshold='v > v_threshold', reset='v = V_reset', 
                                 method='exact', refractory=refractory_time)

# Poisson Inputs für 20 zufällige exzitatorische Neuronen mit 100 Hz Rate
poisson_inputs = PoissonGroup(20, rates=100 * Hz)  # 20 Poisson Inputs
input_synapses = Synapses(poisson_inputs, E, on_pre='v_post += 6.4 * mV')
target_neurons = random.sample(range(N_E), 20)  # Wähle 20 zufällige Zielneuronen
input_synapses.connect(i=range(20), j=target_neurons)

# Synapsen zwischen exzitatorischen Neuronen (5% Wahrscheinlichkeit)
S_E_E = Synapses(E, E, on_pre='v_post += 5.3 * mV')
S_E_E.connect(condition='i != j', p=0.05)  # Keine Selbstverbindungen, 5% Wahrscheinlichkeit

# Synapsen von exzitatorischen zu inhibitorischen Neuronen (5% Wahrscheinlichkeit)
S_E_I = Synapses(E, I, on_pre='v_post += 5.3 * mV')
S_E_I.connect(p=0.05)

# Synapsen von inhibitorischen zu exzitatorischen Neuronen (5% Wahrscheinlichkeit)
S_I_E = Synapses(I, E, on_pre='v_post -= 3.7 * mV')
S_I_E.connect(p=0.05)

# Synapsen zwischen inhibitorischen Neuronen (2% Wahrscheinlichkeit)
S_I_I = Synapses(I, I, on_pre='v_post -= 3.7 * mV')
S_I_I.connect(condition='i != j', p=0.02)  # Keine Selbstverbindungen, 2% Wahrscheinlichkeit

# Aufzeichnung der Spikes
spikemon_E = SpikeMonitor(E)
spikemon_I = SpikeMonitor(I)

# Simulation
run(duration)


# Berechnung der Y- und Z-Verschiebungen für jedes exzitatorische und inhibitorische Neuron
y_shift_exc = np.zeros(N_E)
z_shift_exc = np.zeros(N_E)
y_shift_inh = np.zeros(N_I)
z_shift_inh = np.zeros(N_I)

for i, j in zip(S_E_E.i, S_E_E.j):
    y_shift_exc[i] += 1
    z_shift_exc[j] += 1

for i, j in zip(S_I_I.i, S_I_I.j):
    y_shift_inh[i] += 1
    z_shift_inh[j] += 1

for i, j in zip(S_E_I.i, S_E_I.j):
    y_shift_exc[i] += 1
    z_shift_inh[j] += 1

for i, j in zip(S_I_E.i, S_I_E.j):
    y_shift_inh[i] += 1
    z_shift_exc[j] += 1


# 3D-Visualisierung der Konnektivität
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("3D-Konnektivität der Neuronen")
ax1.set_xlabel("Neuron Index")
ax1.set_ylabel("Source Count (Y-Axis Shift)")
ax1.set_zlabel("Target Count (Z-Axis Shift)")

# Exzitatorische Neuronen in Rot
for i in range(N_E):
    ax1.scatter(i, y_shift_exc[i], z_shift_exc[i], color='red', s=30)
    ax1.text(i, y_shift_exc[i], z_shift_exc[i], f'{i}', size=8, color='black')

# Inhibitorische Neuronen in Blau
for i in range(N_I):
    ax1.scatter(i + N_E, y_shift_inh[i], z_shift_inh[i], color='blue', s=30)
    ax1.text(i + N_E, y_shift_inh[i], z_shift_inh[i], f'{i + N_E}', size=8, color='black')

# Verbindungen zwischen exzitatorischen Neuronen
for i, j in zip(S_E_E.i, S_E_E.j):
    ax1.plot([i, j], [y_shift_exc[i], y_shift_exc[j]], [z_shift_exc[i], z_shift_exc[j]], 'k-', alpha=0.3)

# Verbindungen zwischen inhibitorischen Neuronen
for i, j in zip(S_I_I.i, S_I_I.j):
    ax1.plot([i + N_E, j + N_E], [y_shift_inh[i], y_shift_inh[j]], [z_shift_inh[i], z_shift_inh[j]], 'b-', alpha=0.3)

# Verbindungen von exzitatorischen zu inhibitorischen Neuronen
for i, j in zip(S_E_I.i, S_E_I.j):
    ax1.plot([i, j + N_E], [y_shift_exc[i], y_shift_inh[j]], [z_shift_exc[i], z_shift_inh[j]], 'k-', alpha=0.3)

# Verbindungen von inhibitorischen zu exzitatorischen Neuronen
for i, j in zip(S_I_E.i, S_I_E.j):
    ax1.plot([i+ N_E, j], [y_shift_inh[i], y_shift_exc[j]], [z_shift_inh[i], z_shift_exc[j]], 'b-', alpha=0.3)


# 2D-Konnektivitätsdiagramm
ax2 = fig.add_subplot(122)
ax2.set_title("2D-Synapsenverbindungen")
ax2.set_xlabel("Source Neuron Index")
ax2.set_ylabel("Target Neuron Index")

# Synapsen Scatterplots
scatter_exc_exc = ax2.scatter(S_E_E.i, S_E_E.j, color='red', s=10, label="Excitatory to Excitatory")
scatter_exc_inh = ax2.scatter(S_E_I.i, S_E_I.j + N_E, color='purple', s=10, label="Excitatory to Inhibitory")
scatter_inh_inh = ax2.scatter(S_I_I.i + N_E, S_I_I.j + N_E, color='blue', s=10, label="Inhibitory to Inhibitory")
scatter_inh_exc = ax2.scatter(S_I_E.i + N_E, S_I_E.j, color='purple', s=10, label="Inhibitory to Inhibitory")

# Rasterplot der Spike-Zeiten
fig_spikes = figure()
ax3 = fig_spikes.add_subplot(111)
ax3.set_title("Spike Raster Plot")
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Neuron Index")
ax3.set_xlim([0, duration/ms])
ax3.scatter(spikemon_E.t/ms, spikemon_E.i, color='red', s=2, label="Excitatory")
ax3.scatter(spikemon_I.t/ms, spikemon_I.i + N_E, color='blue', s=2, label="Inhibitory")
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.tight_layout()
plt.show()
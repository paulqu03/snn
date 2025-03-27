import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

pickle_name = 'network.pkl'
network_path = './experiments/inh_fast/current_state/standard_network.pkl'
base_path = './experiments/inh_fast/connectivity_probability_E_I_increase'

# Lade die gespeicherte Netzwerkkonfiguration
with open(network_path, 'rb') as f:
    loaded_config = pickle.load(f)

# Parameter laden
N_E = loaded_config["N_E"]
N_I = loaded_config["N_I"]
v_threshold_E = loaded_config["v_threshold_E"]
v_threshold_I = loaded_config["v_threshold_I"]
V_reset_E = loaded_config["V_reset_E"]
V_reset_I = loaded_config["V_reset_I"]
tau = loaded_config["tau"]
refractory_time = loaded_config["refractory_time"]
duration = loaded_config["duration"]
eqs = loaded_config["eqs"]
N_P = loaded_config["N_P"]
input_rate = loaded_config["input_rate"]
v_post_poisson = loaded_config["v_post_poisson"]
v_post_E_E = loaded_config["v_post_E_E"]
v_post_E_I = loaded_config["v_post_E_I"]
v_post_I_E = loaded_config["v_post_I_E"]
v_post_I_I = loaded_config["v_post_I_I"]
p_E_E = loaded_config["p_E_E"]
p_I_E = loaded_config["p_I_E"]
p_I_I = loaded_config["p_I_I"]
target_neurons = loaded_config["target_neurons"]

p_E_I_values = [0.05, 0.10, 0.15, 0.20]



'''def visualise_connectivity(S):
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
    ylabel('Target neuron index')'''


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Erstellen von exzitatorischen und inhibitorischen Neuronengruppen
E = NeuronGroup(N_E, eqs, threshold='v > v_threshold_E', reset='v = V_reset_E', 
                                 method='exact', refractory=refractory_time)
I = NeuronGroup(N_I, eqs, threshold='v > v_threshold_I', reset='v = V_reset_I', 
                                 method='exact', refractory=refractory_time)


# Poisson Inputs für 20 zufällige exzitatorische Neuronen mit 100 Hz Rate
poisson_inputs = PoissonGroup(N_P, rates= input_rate)  # 20 Poisson Inputs
input_synapses = Synapses(poisson_inputs, E, on_pre='v_post += v_post_poisson')
target_neurons = random.sample(range(N_E), N_P)  # Wähle 20 zufällige Zielneuronen
input_synapses.connect(i=range(N_P), j=target_neurons)


# Synapsen zwischen exzitatorischen Neuronen (5% Wahrscheinlichkeit)
S_E_E = Synapses(E, E, on_pre='v_post += v_post_E_E')
S_E_E.connect(condition='i != j', p=p_E_E)  # Keine Selbstverbindungen, 5% Wahrscheinlichkeit

'''# Synapsen von exzitatorischen zu inhibitorischen Neuronen (5% Wahrscheinlichkeit)
S_E_I = Synapses(E, I, on_pre='v_post += v_post_E_I')
S_E_I.connect(p=p_E_I)'''

# Synapsen von inhibitorischen zu exzitatorischen Neuronen (5% Wahrscheinlichkeit)
S_I_E = Synapses(I, E, on_pre='v_post -= v_post_I_E')
S_I_E.connect(p=p_I_E)

# Synapsen zwischen inhibitorischen Neuronen (2% Wahrscheinlichkeit)
S_I_I = Synapses(I, I, on_pre='v_post -= v_post_I_I')
S_I_I.connect(condition='i != j', p=p_I_I)  # Keine Selbstverbindungen, 2% Wahrscheinlichkeit


# Aufzeichnung der Spikes
spikemon_E = SpikeMonitor(E)
spikemon_I = SpikeMonitor(I)


# Network erstellen und alle Komponenten hinzufügen
net = Network()
net.add(E, I, poisson_inputs, input_synapses, S_E_E, S_I_E, S_I_I, spikemon_E, spikemon_I)
net.store()

for p_E_I in p_E_I_values:
    
    net.restore()

    # Erstelle einen Ordner für den aktuellen Schwellenwert
    folder_name = f"p_E_I_{float(p_E_I)}"
    experiment_path = os.path.join(base_path, folder_name)
    pickle_path = os.path.join(experiment_path, pickle_name)
    create_directory(experiment_path)

    # Synapsen von exzitatorischen zu inhibitorischen Neuronen (5% Wahrscheinlichkeit)
    S_E_I = Synapses(E, I, on_pre='v_post += v_post_E_I')
    S_E_I.connect(p=p_E_I)

    # Dictionary für das Netzwerk erstellen
    network_config = {
        "N_E": N_E,
        "N_I": N_I,
        "v_threshold_E": v_threshold_E,
        "v_threshold_I": v_threshold_I,
        "V_reset_E": V_reset_E,
        "V_reset_I": V_reset_I,
        "tau": tau,
        "refractory_time": refractory_time,
        "duration": duration,
        "eqs": eqs,
        "N_P": N_P,
        "input_rate": input_rate,
        "v_post_poisson": v_post_poisson,
        "v_post_E_E": v_post_E_E,
        "v_post_E_I": v_post_E_I,
        "v_post_I_E": v_post_I_E,
        "v_post_I_I": v_post_I_I,
        "p_E_E": p_E_E,
        "p_E_I": p_E_I,
        "p_I_E": p_I_E,
        "p_I_I": p_I_I,
        "target_neurons": target_neurons
    }

    # Netzwerk mit pickle speichern
    with open(pickle_path, 'wb') as f:
        pickle.dump(network_config, f)
    
    net.add(S_E_I)
    
    # Simulation
    net.run(duration)
    net.remove(S_E_I)


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
    ax1.set_title(f"3D-Konnektivität der Neuronen (p_E_I={float(p_E_I)})")
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
    ax2.set_title(f"2D-Synapsenverbindungen (p_E_I={float(p_E_I)})")
    ax2.set_xlabel("Source Neuron Index")
    ax2.set_ylabel("Target Neuron Index")

    # Synapsen Scatterplots
    scatter_exc_exc = ax2.scatter(S_E_E.i, S_E_E.j, color='red', s=10, label="Excitatory to Excitatory")
    scatter_exc_inh = ax2.scatter(S_E_I.i, S_E_I.j + N_E, color='purple', s=10, label="Excitatory to Inhibitory")
    scatter_inh_inh = ax2.scatter(S_I_I.i + N_E, S_I_I.j + N_E, color='blue', s=10, label="Inhibitory to Inhibitory")
    scatter_inh_exc = ax2.scatter(S_I_E.i + N_E, S_I_E.j, color='purple', s=10, label="Inhibitory to Inhibitory")

    fig.savefig(os.path.join(experiment_path, 'connectivity.png'))
    plt.close(fig)

    # Rasterplot der Spike-Zeiten
    fig_spikes = figure()
    ax3 = fig_spikes.add_subplot(111)
    ax3.set_title(f"Spike Raster Plot (p_E_I={float(p_E_I)})")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Neuron Index")
    ax3.set_xlim([0, duration/ms])
    ax3.scatter(spikemon_E.t/ms, spikemon_E.i, color='red', s=2, label="Excitatory")
    ax3.scatter(spikemon_I.t/ms, spikemon_I.i + N_E, color='blue', s=2, label="Inhibitory")
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_spikes.savefig(os.path.join(experiment_path, 'spike_raster_plot.png'))
    plt.close(fig_spikes)


#plt.tight_layout()
#plt.show()'''
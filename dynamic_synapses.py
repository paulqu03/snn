from brian2 import *
import numpy as np
import os

base_path='./experiments/synapse_compare'

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Netzwerkparameter
v_threshold = 15 * mV
v_reset = 0 * mV
tau = 20 * ms
tau_syn = 500*ms
refractory_time = 5 * ms
duration = 1000 * ms
input_rates=[10, 20, 30, 40]*Hz


results = {"exponential": [], "alpha": [], "biexponential": []}

for rate in input_rates:

    folder_name = f"rate{rate/Hz}hz"
    experiment_path = os.path.join(base_path, folder_name)
    create_directory(experiment_path)

    # Inputneuron (Poisson-Gruppe)
    input_neuron = PoissonGroup(1, rates=rate)
    
    eqs_exponential='''
    dV/dt = -V/tau: volt
    '''

    # Exponential-Synapse
    G_exponential = NeuronGroup(1, eqs_exponential, threshold='V >= v_threshold', reset='V = v_reset', method='exact')
    S_exponential = Synapses(input_neuron, G_exponential,model='''
                                                                dg/dt = -g/tau_syn : volt (clock-driven)
                                                                w : volt
                                                                ''', 
                                                                on_pre='g -= w; V += g')
    S_exponential.connect()
    S_exponential.w= 0.5*mV
    S_exponential.g=10*mV
    M_exponential = StateMonitor(G_exponential, 'V', record=True)

    # Alpha-Synapse
    tau_alpha = 50*ms
    eqs_alpha = '''
    dV/dt = (x-V)/tau : volt
    dx/dt = -x/tau_alpha    : volt
    '''

    G_alpha = NeuronGroup(1, eqs_alpha, threshold='V >= v_threshold', reset='V = v_reset', method='exact')
    S_alpha = Synapses(input_neuron, G_alpha, model='w : volt', on_pre='x += w')
    S_alpha.connect()
    S_alpha.w = 5*mV
    M_alpha = StateMonitor(G_alpha, 'V', record=True)

    # Biexponentiale Synapse
    tau_1 = 5*ms
    tau_2 = 1*ms
    eqs_biexponential = '''
    dV/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-V)/tau_1 : volt
    dx/dt = -x/tau_2 : volt
    '''

    G_biexponential = NeuronGroup(1, eqs_biexponential, threshold='V >= v_threshold', reset='V = v_reset', method='exact')
    S_biexponential = Synapses(input_neuron, G_biexponential, model='w : volt',on_pre='x += w')
    S_biexponential.connect()
    S_biexponential.w = 5*mV
    M_biexponential = StateMonitor(G_biexponential, 'V', record=True)

    run(duration)

    results["exponential"].append(M_exponential.V[0])
    results["alpha"].append(M_alpha.V[0])
    results["biexponential"].append(M_biexponential.V[0])

    time = M_exponential.t/ms

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Exponential-Synapse
    axs[0].plot(time, M_exponential.V[0] / mV, color="blue")
    axs[0].set_title(f"Exponential Synapse (Rate: {rate/Hz} Hz)")
    axs[0].set_ylabel("Membranpotential (mV)")

    # Alpha-Synapse
    axs[1].plot(time, M_alpha.V[0] / mV, color="green")
    axs[1].set_title(f"Alpha Synapse (Rate: {rate/Hz} Hz)")
    axs[1].set_ylabel("Membranpotential (mV)")

    # Biexponentiale Synapse
    axs[2].plot(time, M_biexponential.V[0] / mV, color="red")
    axs[2].set_title(f"Biexponential Synapse (Rate: {rate/Hz} Hz)")
    axs[2].set_xlabel("Zeit (ms)")
    axs[2].set_ylabel("Membranpotential (mV)")

    fig.tight_layout()
    fig.savefig(os.path.join(experiment_path, "membran_potential.png"))
    plt.close(fig)
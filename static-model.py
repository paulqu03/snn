import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

pickle_name = 'network_config.pkl'
network_path = './experiments/decision_simulation/standard_network_config.pkl'
base_path = './experiments/decision_simulation/model_testing/different_values/bg_input_rate'

if os.path.exists(network_path):
    with open(network_path, 'rb') as f:
        loaded_config = pickle.load(f)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 
# Load Pickle Parameter
# -------------------------------------------------------

# Temp Values

N_E1 = 100  # number of neurons in E1
N_E2 = 100  # number of neurons in E2
N_I = 20    # number of neurons in I

eqs_E1 = '''
dv/dt = -v / tau_E1 : volt (unless refractory)
'''
eqs_E2 = '''
dv/dt = -v / tau_E2 : volt (unless refractory)
'''
eqs_I = '''
dv/dt = -v / tau_I : volt (unless refractory)
'''

sigma = 4.0 * Hz            # standard deviation of stimulus input
mu0 = 70.0 * Hz             # stimulus input 0
mu1 = 70.0 * Hz             # stimulus input 1
stim_interval = 50.0 * ms   # stimulus changes every 50 ms
stim_on = 500 * ms          # stimulus onset
stim_off = 1500 * ms        # stimulus offset
runtime = 2000 * ms         # total simulation time

tau_E1 = 20*ms
tau_E2 = 20*ms
tau_I = 10*ms

v_threshold_E1 = 15*mV
v_threshold_E2 = 15*mV
v_threshold_I = 15*mV

v_reset_E1 = 5*mV
v_reset_E2 = 5*mV
v_reset_I = 5*mV

refractory_time_E1 = 0*ms
refractory_time_E2 = 0*ms
refractory_time_I = 0*ms

bg_poisson_rate = 80*Hz

v_post_poisson = 1.4*mV

v_post_stim = 3.8*mV

v_post_E1 = 2.3*mV #
v_post_E2 = 2.3*mV #
v_post_I = 4.5*mV #

v_post_E1_self = 0.5*mV #
v_post_E2_self = 0.5*mV #
v_post_I_self = 0.01*mV

p_P_E1 = 0.05
p_P_E2 = 0.05

p_E1_E1 = 0.40
p_E2_E2 = 0.40
p_I_I = 0.05 

p_E1_I = 0.20
p_E2_I = 0.20
p_I_E2 = 0.40
p_I_E1 = 0.40

# -------------------------------------------------------
# Set Parameters
# -------------------------------------------------------

experiment = [10, 30, 50, 70, 90]*Hz
experiment_name = "input_rate"

# -------------------------------------------------------
# Run simulation and generate plots
# -------------------------------------------------------
for value in experiment:

    folder_name = f"{experiment_name}_{float(value)}Hz"
    experiment_path = os.path.join(base_path, folder_name)
    pickle_path = os.path.join(experiment_path, pickle_name)
    create_directory(experiment_path)
    
    bg_poisson_rate = value

    E1 = NeuronGroup(N_E1, eqs_E1, threshold='v >= v_threshold_E1', reset='v = v_reset_E1', method='euler', refractory=refractory_time_E1)
    E2 = NeuronGroup(N_E2, eqs_E2, threshold='v >= v_threshold_E2', reset='v = v_reset_E2', method='euler', refractory=refractory_time_E2)
    I = NeuronGroup(N_I, eqs_I, threshold='v >= v_threshold_I', reset='v = v_reset_I', method='euler', refractory=refractory_time_I)
    
    E1.v = 'v_threshold_E1-5*mV'
    E2.v = 'v_threshold_E2-5*mV'
    I.v = 'v_threshold_I-5*mV'

    poisson_input_E1 = PoissonGroup(100, rates=bg_poisson_rate)
    poisson_input_E2 = PoissonGroup(100, rates=bg_poisson_rate)


    S_P_E1 = Synapses(poisson_input_E1, E1, on_pre='v += v_post_poisson')
    S_P_E2 = Synapses(poisson_input_E2, E2, on_pre='v += v_post_poisson')

    S_P_E1.connect(p=p_P_E1)
    S_P_E2.connect(p=p_P_E2)

    stiminputE1 = PoissonGroup(20, rates=0*Hz, name='stiminputE1')
    stiminputE2 = PoissonGroup(20, rates=0*Hz, name='stiminputE2')

    mu0 = 60*Hz
    mu1 = 60*Hz

    # Every 50ms a new rate will be set
    stiminputE1.run_regularly(
        "rates = int(t > stim_on and t < stim_off) * "
        "(mu0 + sigma*randn())",
        dt=stim_interval
    )
    stiminputE2.run_regularly(
        "rates = int(t > stim_on and t < stim_off) * "
        "(mu1 + sigma*randn())",
        dt=stim_interval
    )

    C_stimE1 = Synapses(stiminputE1, E1, on_pre='v += v_post_stim', name='C_stimE1', delay=0.5*ms)
    C_stimE1.connect(p=p_P_E1)
    C_stimE2 = Synapses(stiminputE2, E2, on_pre='v += v_post_stim', name='C_stimE2', delay=0.5*ms)
    C_stimE2.connect(p=p_P_E2)

    S_E1_E1 = Synapses(E1, E1, on_pre='v += v_post_E1_self', delay=0.5*ms)
    S_E2_E2 = Synapses(E2, E2, on_pre='v += v_post_E2_self', delay=0.5*ms)
    S_I_I = Synapses(I, I, on_pre='v -= v_post_I_self', delay=0.5*ms)

    S_E1_I = Synapses(E1, I, on_pre='v += v_post_E1', delay=0.5*ms)
    S_E2_I = Synapses(E2, I, on_pre='v += v_post_E2', delay=0.5*ms)
    S_I_E2 = Synapses(I, E2, on_pre='v -= v_post_I', delay=0.5*ms)
    S_I_E1 = Synapses(I, E1, on_pre='v -= v_post_I', delay=0.5*ms)

    S_E1_E1.connect(condition='i != j', p=p_E1_E1)
    S_E2_E2.connect(condition='i != j', p=p_E2_E2)
    S_I_I.connect(condition='i != j', p=p_I_I)

    S_E1_I.connect(p=p_E1_I)
    S_E2_I.connect(p=p_E2_I)
    S_I_E2.connect(p=p_I_E2)
    S_I_E1.connect(p=p_I_E1)

    PopRateMon_E1 = PopulationRateMonitor(E1, name='P_E1')
    PopRateMon_E2 = PopulationRateMonitor(E2, name='P_E2')
    PopRateMon_I = PopulationRateMonitor(I, name='P_I')


    SpikeMon_E1 = SpikeMonitor(E1)
    SpikeMon_E2 = SpikeMonitor(E2)
    SpikeMon_I = SpikeMonitor(I)


    S_E1 = StateMonitor(stiminputE1, 'rates', record=0, dt=1*ms)
    S_E2 = StateMonitor(stiminputE2, 'rates', record=0, dt=1*ms)


    run(runtime)

    # Pickle Save

    bin_size = 15*ms
    
    fig1 = figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    rate_smoothed_E1 = PopRateMon_E1.smooth_rate(window='flat', width=bin_size)/Hz
    ax1.plot(PopRateMon_E1.t/ms, rate_smoothed_E1, label='E1 Rate')

    rate_smoothed_E2 = PopRateMon_E2.smooth_rate(window='flat', width=bin_size)/Hz
    ax1.plot(PopRateMon_E2.t/ms, rate_smoothed_E2, label='E2 Rate')
    
    '''rate_smoothed_I = _PopRateMonI.smooth_rate(window='flat', width=bin_size)/Hz
    ax1.plot(PopRateMon_I.t/ms, rate_smoothed_I, label='I Rate')
    '''

    ax1.set_title(f'Spikerates mit Bin-Size = {bin_size/ms}ms')
    ax1.set_xlabel('Zeit (ms)')
    ax1.set_ylabel('Rate (spikes/s)')
    ax1.legend()

    fig1.savefig(os.path.join(experiment_path, 'spike_activity.png'))
    plt.close(fig1)

    fig2 = figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Spike Raster Plot")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Neuron Index")
    ax2.set_xlim([0, runtime/ms])
    ax2.scatter(SpikeMon_E1.t/ms, SpikeMon_E1.i, color='red', s=2, label="E1")
    ax2.scatter(SpikeMon_E2.t/ms, SpikeMon_E2.i+N_E1, color='blue', s=2, label="E2")
    ax2.scatter(SpikeMon_I.t/ms, SpikeMon_I.i+N_E1+N_E2, color='green', s=2, label="I")
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig2.savefig(os.path.join(experiment_path, 'spike_raster_plot.png'))
    plt.close(fig2)

    fig3 = figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(S_E1.t / ms, S_E1.rates[0] / Hz, color='darkred')
    ax3.plot(S_E2.t / ms, S_E2.rates[0] / Hz, color='darkblue')
    ax3.set(ylabel='Input (Hz)', xlabel='Time (ms)')
    fig3.savefig(os.path.join(experiment_path, 'stimulus.png'))
    plt.close(fig3)
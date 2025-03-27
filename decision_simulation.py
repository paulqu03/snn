import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

#############################################################
# Load parameters
#############################################################

pickle_name = 'network_config.pkl'
network_path = './experiments/decision_simulation/standard_network_config.pkl'
base_path = './experiments/decision_simulation/model_testing/different_values/input_rate'

if os.path.exists(network_path):
    with open(network_path, 'rb') as f:
        loaded_config = pickle.load(f)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
'''
#############################################################
# Get parameters
#############################################################

N_E1 = loaded_config['N_E1']    # N_E1 = 100
N_E2 = loaded_config['N_E2']    # N_E2 = 100
N_I = loaded_config['N_I']    # N_I = 20
'''
#eqs_E1 = loaded_config['eqs_E1']    # eqs_E1 = '''dv/dt = -v / tau_E1 : volt (unless refractory)'''
#eqs_E2 = loaded_config['eqs_E2']    # eqs_E2 = '''dv/dt = -v / tau_E2 : volt (unless refractory)'''
#eqs_I = loaded_config['eqs_I']    # eqs_I = '''dv/dt = -v / tau_I : volt (unless refractory)'''
'''
tau_E1 = loaded_config['tau_E1']    # tau_E1 = 20*ms
tau_E2 = loaded_config['tau_E2']    # tau_E2 = 20*ms
tau_I = loaded_config['tau_I']      # tau_I  =  20*ms

v_threshold_E1 = loaded_config['v_threshold_E1']    # v_threshold_E1 = 15*mV
v_threshold_E2 = loaded_config['v_threshold_E2']    # v_threshold_E2 = 15*mV
v_threshold_I = loaded_config['v_threshold_I']      # v_threshold_I = 10*mV

v_reset_E1 = loaded_config['v_reset_E1']    # v_reset_E1 = 0*mV
v_reset_E2 = loaded_config['v_reset_E2']    # v_reset_E2 = 0*mV
v_reset_I = loaded_config['v_reset_I']      # v_reset_I = 0*mV

refractory_time_E1 = loaded_config['refractory_time_E1']    # refractory_time_E1 = 5*ms
refractory_time_E2 = loaded_config['refractory_time_E2']    # refractory_time_E2 = 5*ms
refractory_time_I = loaded_config['refractory_time_I']      # refractory_time_I = 5*ms

N_P_E1 = loaded_config['N_P_E1']    # N_P_E1 = 40
N_P_E2 = loaded_config['N_P_E2']    # N_P_E2 = 40

input_rates_E1 = loaded_config['input_rates_E1']    # input_rates_E1 = 50*Hz
input_rates_E2 = loaded_config['input_rates_E2']    # input_rates_E2 = 100*Hz

v_post_poisson_E1 = loaded_config['v_post_poisson_E1']  # v_post_poisson_E1 = 6*mV
v_post_poisson_E2 = loaded_config['v_post_poisson_E2']  # v_post_poisson_E2 = 6*mV

v_post_E1 = loaded_config['v_post_E1']      # v_post_E1 = 8*mV
v_post_E2 = loaded_config['v_post_E2']      # v_post_E2 = 8*mV
v_post_I = loaded_config['v_post_I']        # v_post_I = 4*mV

v_post_E1_self = loaded_config['v_post_E1_self']    # v_post_E1_self = 5*mV
v_post_E2_self = loaded_config['v_post_E2_self']    # v_post_E2_self = 5*mV
v_post_I_self = loaded_config['v_post_I_self']      # v_post_I_self = 2.5*mV

p_E1_E1 = loaded_config['p_E1_E1']      # p_E1_E1 = 0.05
p_E2_E2 = loaded_config['p_E2_E2']      # p_E2_E2 = 0.05
p_I_I = loaded_config['p_I_I']          # p_I_I = 0.05

p_E1_I = loaded_config['p_E1_I']      # p_E1_I = 0.10
p_E2_I = loaded_config['p_E2_I']      # p_E2_I = 0.10
p_I_E2 = loaded_config['p_I_E2']      # p_I_E2 = 0.15
p_I_E1 = loaded_config['p_I_E1']      # p_I_E1 = 0.15'''


#Temp

N_E1 = 100
N_E2 = 100
N_I = 20

eqs_E1 = '''
dv/dt = -v / tau_E1 : volt (unless refractory)
'''
eqs_E2 = '''
dv/dt = -v / tau_E2 : volt (unless refractory)
'''
eqs_I = '''
dv/dt = -v / tau_I : volt (unless refractory)
'''


tau_E1 = 20*ms
tau_E2 = 20*ms
tau_I = 10*ms

v_threshold_E1 = 15*mV
v_threshold_E2 = 15*mV
v_threshold_I = 15*mV

v_reset_E1 = 0*mV
v_reset_E2 = 0*mV
v_reset_I = 0*mV

refractory_time_E1 = 0*ms
refractory_time_E2 = 0*ms
refractory_time_I = 0*ms

input_rate_E1 = 90*Hz
input_rate_E2 = 100*Hz

v_post_poisson_E1 = 3.3*mV
v_post_poisson_E2 = 3.3*mV

v_post_E1 = 2.5*mV #
v_post_E2 = 2.5*mV #
v_post_I = 3.5*mV #

v_post_E1_self = 1.6*mV #
v_post_E2_self = 1.6*mV #
v_post_I_self = 0.01*mV

p_P_E1 = 0.05
p_P_E2 = 0.05

p_E1_E1 = 0.12
p_E2_E2 = 0.12
p_I_I = 0.05 

p_E1_I = 0.20
p_E2_I = 0.20
p_I_E2 = 0.25
p_I_E1 = 0.25


#############################################################
# Set paramters
#############################################################

duration = 1000*ms
experiment = [50, 60, 70, 80, 90, 100]*Hz
experiment_name = "input_rate"

#############################################################
# Run simulation and generate plots
#############################################################

for value in experiment:

    folder_name = f"{experiment_name}_{float(value)}Hz"
    experiment_path = os.path.join(base_path, folder_name)
    pickle_path = os.path.join(experiment_path, pickle_name)
    create_directory(experiment_path)
    
    input_rate_E1 = value

    E1 = NeuronGroup(N_E1, eqs_E1, threshold='v >= v_threshold_E1', reset='v = v_reset_E1', method='exact', refractory=refractory_time_E1)
    E2 = NeuronGroup(N_E2, eqs_E2, threshold='v >= v_threshold_E2', reset='v = v_reset_E2', method='exact', refractory=refractory_time_E2)
    I = NeuronGroup(N_I, eqs_I, threshold='v >= v_threshold_I', reset='v = v_reset_I', method='exact', refractory=refractory_time_I)
    
    E1.v = 'rand()*v_threshold_E1'
    E2.v = 'rand()*v_threshold_E2'
    I.v = 'rand()*v_threshold_I'

    poisson_input_E1 = PoissonGroup(20, rates=input_rate_E1)
    poisson_input_E2 = PoissonGroup(20, rates=input_rate_E2)


    S_P_E1 = Synapses(poisson_input_E1, E1, on_pre='v += v_post_poisson_E1')
    S_P_E2 = Synapses(poisson_input_E2, E2, on_pre='v += v_post_poisson_E2')

    S_P_E1.connect(p=p_P_E1)
    S_P_E2.connect(p=p_P_E2)

    S_E1_E1 = Synapses(E1, E1, on_pre='v += v_post_E1_self')
    S_E2_E2 = Synapses(E2, E2, on_pre='v += v_post_E2_self')
    S_I_I = Synapses(I, I, on_pre='v -= v_post_I_self')

    S_E1_I = Synapses(E1, I, on_pre='v += v_post_E1')
    S_E2_I = Synapses(E2, I, on_pre='v += v_post_E2')
    S_I_E2 = Synapses(I, E2, on_pre='v -= v_post_I')
    S_I_E1 = Synapses(I, E1, on_pre='v -= v_post_I')

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


    run(duration)

    network_config = {
        'N_E1': N_E1,
        'N_E2': N_E2,
        'N_I': N_I,
        'eqs_E1': eqs_E1,
        'eqs_E2': eqs_E2,
        'eqs_I': eqs_I,
        'tau_E1': tau_E1,
        'tau_E2': tau_E2,
        'tau_I': tau_I,
        'v_threshold_E1': v_threshold_E1,
        'v_threshold_E2': v_threshold_E2,
        'v_threshold_I': v_threshold_I,
        'v_reset_E1': v_reset_E1,
        'v_reset_E2': v_reset_E2,
        'v_reset_I': v_reset_I,
        'refractory_time_E1': refractory_time_E1,
        'refractory_time_E2': refractory_time_E2,
        'refractory_time_I': refractory_time_I,
        'input_rate_E1': input_rate_E1,
        'input_rate_E2': input_rate_E2,
        'v_post_poisson_E1' : v_post_poisson_E1,
        'v_post_poisson_E2' : v_post_poisson_E2,
        'v_post_E1': v_post_E1,
        'v_post_E2': v_post_E2,
        'v_post_I': v_post_I,
        'v_post_E1_self': v_post_E1_self,
        'v_post_E2_self': v_post_E2_self,
        'v_post_I_self': v_post_I_self,
        'p_E1_E1': p_E1_E1,
        'p_E2_E2': p_E2_E2,
        'p_I_I': p_I_I,
        'p_E1_I': p_E1_I,
        'p_E2_I': p_E2_I,
        'p_I_E2': p_I_E2,
        'p_I_E1': p_I_E1,
    }

    # Netzwerk mit pickle speichern
    with open(pickle_path, 'wb') as f:
        pickle.dump(network_config, f)

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
    ax2.set_xlim([0, duration/ms])
    ax2.scatter(SpikeMon_E1.t/ms, SpikeMon_E1.i, color='red', s=2, label="E1")
    ax2.scatter(SpikeMon_E2.t/ms, SpikeMon_E2.i+N_E1, color='blue', s=2, label="E2")
    ax2.scatter(SpikeMon_I.t/ms, SpikeMon_I.i+N_E1+N_E2, color='green', s=2, label="I")
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig2.savefig(os.path.join(experiment_path, 'spike_raster_plot.png'))
    plt.close(fig2)
    

    
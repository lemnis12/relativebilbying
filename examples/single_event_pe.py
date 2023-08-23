import numpy as np 
import bilby 
from relativebilbying import separate_l_likelihood
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim



### setting up rundir

outdir = 'example_single_event_run'
label = 'trial_0'
bilby.core.utils.setup_logger(outdir=outdir, label=label)
seed = 2023
total_err = 0.01

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)



injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters['mass_2'])
injection_parameters['mass_ratio'] = injection_parameters['mass_2']/injection_parameters['mass_1']

fiducial_parameters = injection_parameters.copy()

percentage_error = 10.

print('percentage_error', percentage_error)
np.random.seed(seed)
print('setting the seed to ', seed)
random_error = 1+np.random.uniform(-percentage_error, percentage_error, 1)[0]/100. 

print('random error chose', random_error)
test_parameters = injection_parameters.copy()
test_parameters['chirp_mass'] = injection_parameters['chirp_mass']*random_error
test_parameters['mass_ratio'] = injection_parameters['mass_ratio']*random_error

mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(test_parameters['chirp_mass'], test_parameters['mass_ratio'])

test_parameters['mass_1'] = mass_1
test_parameters['mass_2'] = mass_2

print('\n')
print('test_parameters', test_parameters) 

print('\n')
print('injection_parametes', injection_parameters)
################################################
############ some arguments ###################
waveform_approximant = 'IMRPhenomXPHM'
minimum_frequency = 20.
sampling_frequency = 2048.
reference_frequency = 50.
approx_duration = bilby.gw.utils.calculate_time_to_merger(minimum_frequency, injection_parameters['mass_1'], injection_parameters['mass_2'], safety = 1.2)
duration = np.ceil(approx_duration + 4.)

if duration>5:
    sampling_frequency = 4096.

mode_array = mode_array = [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]]

#################################################
##################################################




waveform_arguments = dict(waveform_approximant = waveform_approximant, reference_frequency = reference_frequency, sampling_frequency = sampling_frequency,
                         minimum_frequency = minimum_frequency,  maximum_frequency = sampling_frequency/2., mode_array = mode_array)

waveform_generator = bilby.gw.WaveformGenerator(duration = duration,
                                                            sampling_frequency= sampling_frequency,
                                                            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments = waveform_arguments)
                

ifos = bilby.gw.detector.InterferometerList(["L1", "H1", "V1"])

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration + 2)


ifos.inject_signal(parameters = injection_parameters, waveform_generator = waveform_generator)


bilby_prior_dict = bilby.gw.prior.BBHPriorDict()

prior = {}



min_mc = np.max([5, injection_parameters['chirp_mass']-5.])
max_mc = injection_parameters['chirp_mass']+5.

prior['chirp_mass'] = bilby.core.prior.Uniform(name = 'chirp_mass', latex_label = '$M_{c}$', minimum = min_mc, maximum = max_mc, unit = '$M_{\\odot}$')
prior['mass_ratio'] = bilby.gw.prior.Uniform(minimum = .1, maximum =  1., name = 'mass_ratio')


prior['a_1'] = bilby_prior_dict['a_1']
prior['a_2'] = bilby_prior_dict['a_2']
prior['phi_12'] = bilby_prior_dict['phi_12']
prior['phi_jl'] = bilby_prior_dict['phi_jl']
prior['tilt_1'] = bilby_prior_dict['tilt_1']
prior['tilt_2'] = bilby_prior_dict['tilt_2']


prior['ra'] = bilby_prior_dict['ra']
prior['dec'] = bilby_prior_dict['dec']

prior['luminosity_distance'] = bilby_prior_dict['luminosity_distance']

prior['psi'] = bilby_prior_dict['psi']
prior['phase'] = bilby_prior_dict['phase']
prior['theta_jn'] = bilby_prior_dict['theta_jn']

prior['geocent_time'] = bilby.gw.prior.Uniform(minimum = injection_parameters['geocent_time']-.1, maximum = injection_parameters['geocent_time']+.1, name = 'geocent_time')


lal_max_f = separate_l_likelihood.lal_f_max(separate_l_likelihood.relby_conversion(injection_parameters, waveform_arguments))
print('maximum freq of lal', lal_max_f)
############################


likelihood = separate_l_likelihood.RelativeBinningHOM(ifos, injection_parameters, test_parameters, prior, lal_max_f, mode_array, waveform_arguments, total_err, rebin_c = False, grid_choice = 'step')



binning_info = likelihood.binning_info
number_of_bins = binning_info[1]

print('number of bins relative binning made:', number_of_bins)
print('bilby bins:', len(np.arange(minimum_frequency, .5*sampling_frequency, 1/duration)))

np.savez('{}/binning_info_label.npz'.format(outdir, label), bin_edge_freqs = binning_info[0])

result = bilby.run_sampler(likelihood = likelihood, priors = prior, npool = 16, verbose = True,
     sampler = 'dynesty', nlive=4096, outdir = outdir, label = label, 
        naccept = 60, check_point_plot = True, check_point_delta_t = 1800,
 injection_parameters=injection_parameters,       print_method = 'interval-60', sample = 'acceptance-walk')


injection_parameters.pop('mass_1')
injection_parameters.pop('mass_2')
result.plot_corner(priors = True)



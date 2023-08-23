import sys
import numpy as np 
import bilby 
from relativebilbying import joint_pe_likelihood

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim


outdir = 'example_lensed_event_joint_pe'
label = 'trial_0'
seed_1 = 150914
seed_2 = 140915
total_err = 0.01

bilby.core.utils.setup_logger(outdir=outdir, label=label)

################################################
######### injection parameters and test parameters. Test parameters are used to choose bins ##############


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
    n_phase_1  = 0.,
    delta_t_21 = 3600*4,
    delta_n_21 = 0., 
    mu_21 = 1.4
)

######### mu_21 = (dl2/dl1)^2 ###### 




print('injection_parameters', injection_parameters)
injection_parameters['n_phase_1'] = injection_parameters['n_phase_1']*np.pi
injection_parameters['delta_n_21'] = injection_parameters['delta_n_21']*np.pi #- injection_parameters['n_phase_1']

injection_parameters['mass_ratio'] = injection_parameters['mass_2']/injection_parameters['mass_1']
injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'], 
                        injection_parameters['mass_2'])

percentage_error = 10.

print('percentage_error', percentage_error)

random_error = 1+np.random.uniform(-percentage_error, percentage_error, 1)[0]/100. 

print('random error chose', random_error)
test_parameters = injection_parameters.copy()
test_parameters['chirp_mass'] = injection_parameters['chirp_mass']*random_error
test_parameters['mass_ratio'] = injection_parameters['mass_ratio']*random_error

mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(test_parameters['chirp_mass'], test_parameters['mass_ratio'])

test_parameters['mass_1'] = mass_1
test_parameters['mass_2'] = mass_2

print('\n\n\n\n')
print('test_parameters', test_parameters) 

print('injection_parametes', injection_parameters)


fiducial_model = {'image_1':joint_pe_likelihood.make_lensed_injection(injection_parameters, 1), 'image_2': joint_pe_likelihood.make_lensed_injection(injection_parameters, 2)}
test_model = {'image_1':joint_pe_likelihood.make_lensed_injection(test_parameters, 1), 'image_2': joint_pe_likelihood.make_lensed_injection(test_parameters, 2)}


################################################
############ some arguments ###################
waveform_approximant = 'IMRPhenomXPHM'
minimum_frequency = 20.
reference_frequency = 50.
approx_duration = bilby.gw.utils.calculate_time_to_merger(minimum_frequency, injection_parameters['mass_1'], injection_parameters['mass_2'], safety = 1.2)
duration = np.ceil(approx_duration + 4.)

if duration < 5:
    sampling_frequency = 2048.
else:
    sampling_frequency = 4096.

mode_array = mode_array = [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]]

print('total_err', total_err)
#################################################
##################################################




waveform_arguments = dict(waveform_approximant = waveform_approximant, reference_frequency = reference_frequency, sampling_frequency = sampling_frequency,
                         minimum_frequency = minimum_frequency,  maximum_frequency = sampling_frequency/2., mode_array = mode_array)

waveform_generator = bilby.gw.WaveformGenerator(duration = duration,
                                                            sampling_frequency= sampling_frequency,
                                                            frequency_domain_source_model = joint_pe_likelihood.lensed_lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments = waveform_arguments)


interferometers_image_1 = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in interferometers_image_1:
    ifo.minimum_frequency = minimum_frequency


np.random.seed(seed_1)
interferometers_image_1.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=(fiducial_model['image_1']["geocent_time"] - duration + 2.))

interferometers_image_1.inject_signal(parameters = fiducial_model['image_1'], waveform_generator = waveform_generator)




np.random.seed(seed_2)
interferometers_image_2 = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

for ifo in interferometers_image_2:
    ifo.minimum_frequency = minimum_frequency




interferometers_image_2.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=(fiducial_model['image_2']["geocent_time"] - duration + 2.))

interferometers_image_2.inject_signal(parameters = fiducial_model['image_2'], waveform_generator = waveform_generator)

prior = bilby.gw.prior.BBHPriorDict()

min_mc = np.max([5, injection_parameters['chirp_mass']-5.])
max_mc = injection_parameters['chirp_mass']+5.

prior['chirp_mass'] = bilby.core.prior.Uniform(name = 'chirp_mass', latex_label = '$M_{c}$', minimum = min_mc, maximum = max_mc, unit = '$M_{\\odot}$')
prior['mass_ratio'] = bilby.core.prior.Uniform(name = 'mass_ratio', latex_label = '$q$', minimum = 0.1, maximum = 1., unit = '$M_{\\odot}$')


prior['mass_1'].minimum = 1.#max(1, injection_parameters['mass_1']-10)
prior['mass_2'].minimum = 1.#max(1, injection_parameters['mass_2']-10)


prior['mass_1'].maximum = 200.
prior['mass_2'].maximum = 200.

 
prior['luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(name = 'luminosity_distance',
                                     minimum = 100., maximum = 5000, latex_label = '$D_{l}$')
prior['n_phase_1'] = bilby.core.prior.Uniform(0, np.pi)
prior['delta_t_21'] = bilby.core.prior.Uniform(injection_parameters['delta_t_21']-0.1, injection_parameters['delta_t_21']+0.1)
prior['delta_n_21'] = bilby.core.prior.Uniform(0, np.pi)
prior['mu_21'] = bilby.core.prior.Uniform(0.1, 10)
prior['geocent_time'] = bilby.core.prior.Uniform(minimum = injection_parameters['geocent_time'] - 0.1,
                                                                                                 maximum = injection_parameters['geocent_time'] + 0.1,
                                                                                                 name = 'geocent_time', latex_label = '$t_c$', unit = '$s$')


prior['mu_21'] = bilby.core.prior.Uniform(1, 10)



ifos_dict = {'image_1': interferometers_image_1, 'image_2': interferometers_image_2}

relby_likelihood = joint_pe_likelihood.JointPELikelihood(ifos_dict, fiducial_model=fiducial_model, test_model=test_model, priors=prior, mode_array=mode_array, waveform_arguments=waveform_arguments, 
                                                                                    total_err=total_err)



result = bilby.run_sampler(likelihood = relby_likelihood, priors = prior, npool = 16, verbose = True,
     sampler = 'dynesty', nlive=2048, outdir = outdir, label = label,
        naccept = 60, check_point_plot = True, check_point_delta_t = 1800,
   injection_parameters = injection_parameters,  print_method = 'interval-60', sample = 'acceptance-walk')


result.plot_corner(priors = True)


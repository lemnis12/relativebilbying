"""
Example test run for relative binning for a time domain 
waveform
"""

import bilby 
import numpy as np
import matplotlib.pyplot as plt
from relativebilbying import relativebinninglikelihood as rel_bin_likeli

# set the injection parameters (also used for the model here)
injection_parameters = dict(mass_1 = 36, mass_2 = 29, 
                            a_1=0.4, a_2=0.3, tilt_1=0.5,
                            tilt_2=1.0, phi_12=1.7,
                            phi_jl=0.3,
                            luminosity_distance =2000, theta_jn = 2.9, ra = 2.7, dec = 0.8, 
                            geocent_time = 1249852257.01182, psi = 0.4,
                            phase = 0.4)

# setup usual stuff
duration = 4.
sampling_frequency = 1024.
f_min = 20.
f_max = 512.
f_ref = 20.

# make the WF and inject the signal 
waveform_arguments = dict(waveform_approximant = 'SpinTaylorT5',
                          reference_frequency = f_ref,
                          minimum_frequency = f_min,
                          maximum_frequency = f_max)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration,
                                                sampling_frequency = sampling_frequency,
                                                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
                                                waveform_arguments = waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency,
                                     duration = duration,
                                     start_time = injection_parameters['geocent_time'] - duration + 1)
ifos.inject_signal(waveform_generator = waveform_generator,
                   parameters = injection_parameters)

# setup the priors
priors = bilby.gw.prior.BBHPriorDict()
priors['geocent_time'] = bilby.core.prior.Uniform(minimum = injection_parameters['geocent_time'] - 0.1,
                                                  maximum = injection_parameters['geocent_time'] + 0.1, 
                                                  name = 'geocent_time', latex_label = '$t_c$')

priors['mass_ratio'] = bilby.core.prior.Uniform(minimum = 0.01,
                                                maximum = 1, 
                                                name = 'mass_ratio',
                                                latex_label = '$q$')
priors['chirp_mass'] = bilby.core.prior.Uniform(minimum = 5.,
                                                maximum = 80.,
                                                name = 'chirp_mass',
                                                latex_label = '$M_c$')
priors['mass_1'] = bilby.core.prior.Constraint(name = 'mass_1',
                                               minimum = 5.,
                                               maximum = 100.)
priors['mass_2'] = bilby.core.prior.Constraint(name = 'mass_2',
                                               minimum = 5.,
                                               maximum = 100.)

# initiate the likelihood

likelihood = rel_bin_likeli.TimeDomainRelativeBinning(ifos, waveform_generator,
                                                      injection_parameters,
                                                      priors)


result = bilby.run_sampler(likelihood = likelihood, priors = priors,
                           sampler = 'pymultinest', injection_parameters = injection_parameters,
                           npoints = 2048, outdir = 'Outdir_TD_SpinTaylorT5', 
                           label = 'TD_SpinTaylorT5')
result.plot_corner()
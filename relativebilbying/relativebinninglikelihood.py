import bilby
from bilby.core.likelihood import Likelihood
import numpy as np
from bilby.gw.utils import noise_weighted_inner_product
import lal
import lalsimulation as lalsim

def get_binned_detector_response(ifos, waveform_polarizations, parameters, idxs):
    """
    Function computing the detector responsed for the frequency indices
    computed in the relative binning. This helps obtaining an additional
    speed wrt to the injection for all points and then selecting only the
    good ones.

    ARGS:
    -----
    - ifos: bilby interferometer list. The interferometes for which 
            the detector response should be computed
    - waveform_polarization: dict of arrays. The waveform polarizations 
                             under consideration
    - parameters: dict. The parameters under consideration. 

    RETURNS:
    --------
    - signals: dict. The signal observed in the detectors for the 
               specified polarization-parameters combination
    """

    signals = {}
    
    for ifo in ifos:
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = ifo.antenna_response(parameters['ra'], parameters['dec'],
                                                parameters['geocent_time'],
                                                parameters['psi'], mode)
            signal[mode] = waveform_polarizations[mode]*det_response

        signal_ifo = sum(signal.values())

        # compute the time shifts
        time_shift = ifo.time_delay_from_geocenter(parameters['ra'], parameters['dec'],
                                                   parameters['geocent_time'])
        dt_geocent = parameters['geocent_time'] - ifo.strain_data.start_time
        dt = dt_geocent + time_shift

        # apply the time shift 
        signal_ifo = signal_ifo * np.exp(-1j * 2 * np.pi * dt * ifo.strain_data.frequency_array[idxs])

        # apply the calibration (used only for real data)
        signal_ifo *= ifo.calibration_model.get_calibration_factor(ifo.strain_data.frequency_array[idxs],
                                                                   prefix = 'recalib_{}_'.format(ifo.name),
                                                                   **parameters)
        signals[ifo.name] = signal_ifo

    return signals

def get_binned_polarizations(parameters, waveform_generator, fs):
    """
    Function computing the polarizations for the parameters at the
    frequencies defined in the relative binning scheme

    ARGS:
    -----
    - parameters: dict, dictionary with the parameters for which 
                  the polarizations should be computed
    - waveform_generator: bilby.gw.WaveformGenerator object. Waveform
                          generator used for the run 
    - fs: np.array, array of frequencies for which the polarizations
          should be computed

    Returns:
    --------
    - waveform_polarizations: keys: plus, cross. Each entry corresponds
                              to one of the polarizations 
    """

    if 'lambda_1' not in parameters.keys() and 'lambda_tilde' not in parameters.keys(): #BBH case
        params, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters)
        params['lambda_1'] = 0.
        params['lambda_2'] = 0.
    else:
        params, _ = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(parameters)

    # convert parameters appropriately for lal
    lum_dist = params['luminosity_distance'] * 1e6 * bilby.core.utils.parsec
    m1 = params['mass_1'] * bilby.core.utils.solar_mass
    m2 = params['mass_2'] * bilby.core.utils.solar_mass

    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(theta_jn = params['theta_jn'],
                                                                                          phi_jl = params['phi_jl'],
                                                                                          tilt_1 = params['tilt_1'],
                                                                                          tilt_2 = params['tilt_2'],
                                                                                          phi_12 = params['phi_12'],
                                                                                          a_1 = params['a_1'],
                                                                                          a_2 = params['a_2'],
                                                                                          mass_1 = m1, mass_2 = m2,
                                                                                          reference_frequency = waveform_generator.waveform_arguments['reference_frequency'],
                                                                                          phase = params['phase'])
    # make info for lal format
    fs_seq = lal.CreateREAL8Sequence(len(fs))
    fs_seq.data = fs

    lal_pars = lal.CreateDict()

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(lal_pars,
                                                      int(waveform_generator.waveform_arguments['pn_spin_order']))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(lal_pars,
                                                       int(waveform_generator.waveform_arguments['pn_tidal_order']))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(lal_pars,
                                                       int(waveform_generator.waveform_arguments['pn_phase_order']))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lal_pars,
                                                           int(waveform_generator.waveform_arguments['pn_amplitude_order']))
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, 
                                                       params['lambda_1'])
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, 
                                                       params['lambda_2'])

    # make the polarizations at the correct frequencies
    hpl, hcr = bilby.gw.utils.lalsim_SimInspiralChooseFDWaveformSequence(params['phase'],
                              m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, 
                              waveform_generator.waveform_arguments['reference_frequency'],
                              lum_dist, iota, lal_pars,
                              waveform_generator.waveform_arguments['waveform_approximant'],
                              fs_seq)

    return {'plus' : hpl.data.data, 'cross' : hcr.data.data}


class RelativeBinning(Likelihood):
    """
    Class encoding the likelihood for relative binning compatible
    with bilby
    """

    def __init__(self, interferometers, waveform_generator, model, priors, delta = 0.03):
        """
        Initialization of the class.

        ARGS:
        -----
        - interferometers: list of interferometers that contain the signal
        - waveform_generator: the bilby waveform generator object that should
                              be used to setup the run
        - model: dictionary with the model parameters to be used
                 for the analysis
        - priors: the priors used for the different parameters
        - delta: the precision factor to be used in the run. Default is 0.03
        """

        super(RelativeBinning, self).__init__(dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.priors = priors
        self.model = model 

        # retrieve some useful information for the rest of the run 
        if 'minimum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_min = self.waveform_generator.waveform_arguments['minimum_frequency']
        else:
            self.f_min = 20.
        if 'maximum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_max = self.waveform_generator.waveform_arguments['maximum_frequency']
        else:
            self.f_max = self.waveform_generator.sampling_frequency/2. # take nyquist
        if 'pn_spin_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_spin_order'] = -1
        if 'pn_phase_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_phase_order'] = -1
        if 'pn_tidal_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_tidal_order'] = -1
        if 'pn_amplitude_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_amplitude_order'] = 0.

        self.duration = self.waveform_generator.duration

        self.make_relative_binning_information(delta = delta)
        # dictionary to be filled by the sampler
        self.parameters = {}

        # additional speed up
        self.noise_log_l = self.noise_log_likelihood()

    @property
    def priors(self):
        return self._prior
    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        else:
            self._prior = None

    def make_relative_binning_information(self, delta = 0.03):
        """
        Function computing the summary data for the relative binning

        For the relative binning scheme, we follow the method given in 
        https://arxiv.org/pdf/1806.08792.pdf

        ARGS:
        -----
        - delta: precision factor to be used when constructing the bins
        """

        # make the model data
        model_polas = self.waveform_generator.frequency_domain_strain(parameters = self.model)
        models = dict()
        for ifo in self.interferometers:
            models[ifo.name] = ifo.get_detector_response(model_polas, self.model)
        
        # make array of frequencies
        f_nt = np.linspace(self.f_min, self.f_max, 10000)

        # array of powers coming in the different PN corrections
        ga = np.array([-5./3, -2./3, 1., 5./3, 7./3])

        # compute the coefficient from https://arxiv.org/pdf/1806.08792.pdf
        dalpha = 2 * np.pi / np.abs(self.f_min**ga - self.f_max**ga)
        dphi = np.sum([np.sign(g) * d * f_nt**g for d, g in zip(dalpha, ga)], axis = 0)
        dphi -= dphi[0]

        # construct the frequency bins
        nbin = int(np.ceil(dphi[-1] / delta))
        dphi_grid = np.linspace(dphi[0], dphi[-1], nbin+1)
        self.fbin = np.interp(dphi_grid, dphi, f_nt)
        # set the bin edges to values for which we have template values
        self.fbin_idxs = np.unique(np.argmin(np.abs(self.waveform_generator.frequency_array[:, np.newaxis] - self.fbin), axis = 0))


        self.fbin = self.waveform_generator.frequency_array[self.fbin_idxs]

        self.Nbin = len(self.fbin) - 1 # number of bins
        self.fm = (self.fbin[1:] + self.fbin[:-1])/2.
        self.binwidths = self.fbin[1:] - self.fbin[:-1] # binwidths

        # compute the summary data
        self.A0, self.A1, self.B0, self.B1 = dict(), dict(), dict(), dict()
        for ifo in self.interferometers:
            d = ifo.strain_data.frequency_domain_strain.copy()
            h = models[ifo.name].copy()
            psd = ifo.power_spectral_density_array.copy()
            fs = ifo.strain_data.frequency_array

            self.A0[ifo.name] = (4./self.duration)*np.array([np.sum(d[self.fbin_idxs[b]:self.fbin_idxs[b+1]]*\
                                np.conj(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])/psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])
                                for b in range(self.Nbin)])
            self.A1[ifo.name] = (4./self.duration)*np.array([np.sum((d[self.fbin_idxs[b]:self.fbin_idxs[b+1]]*\
                                 np.conj(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])/psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])*\
                                 (fs[self.fbin_idxs[b]:self.fbin_idxs[b+1]] - self.fm[b]))
                                 for b in range(self.Nbin)])
            self.B0[ifo.name] = (4./self.duration)*np.array([np.sum((np.abs(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])**2)/\
                                 psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])
                                 for b in range(self.Nbin)])
            self.B1[ifo.name] = (4./self.duration)*np.array([np.sum(((np.abs(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])**2)/\
                                 psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])*(fs[self.fbin_idxs[b]:self.fbin_idxs[b+1]] - self.fm[b]))
                                 for b in range(self.Nbin)])

        # keep the model only at the correct points
        self.h0 = dict()
        #self.idxs_zeros = dict()
        for ifo in models.keys():
            self.h0[ifo] = models[ifo][self.fbin_idxs]
            # store indices where h0 is zero to avoid nans later.
            # this is mostly useful when the maximum frequency is too high
            # wrt the maximum frequency reached by the signal
            idxs_zeros = np.where((self.h0[ifo] == 0.))[0] # store indices where h0 is zero to avoid nans later.
            self.h0[ifo][idxs_zeros] = np.inf
        


    def noise_log_likelihood(self):
        """
        Fuction computing the noise log likelihood. 
        This is the same as for the usual runs
        """
        log_l = 0.

        for ifo in self.interferometers:
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask], 
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))

    def log_likelihood_ratio(self):
        """
        Computes the likelihood ratio using the relative binning scheme
        of https://arxiv.org/pdf/1806.08792.pdf.

        This uses the pre-computed summary data to evaluate the 
        likelihood.

        It also uses a custom waveform generator in order to compute the
        data only at the relevant point, saving addtional time
        """
        # first need to get the polarizations
        polas = get_binned_polarizations(self.parameters, self.waveform_generator, self.fbin)
        h = get_binned_detector_response(self.interferometers, polas, self.parameters,
                                         self.fbin_idxs)
        
        d_inner_h = 0
        h_inner_h = 0

        for ifo in h.keys():
            r = h[ifo]/self.h0[ifo]
            #r[self.idxs_zeros[ifo]] = 0.

            rpl = r[1:]
            rmin = r[:-1]

            r0 = (rpl + rmin)/2.
            r1 = (rpl - rmin)/self.binwidths

            d_inner_h += np.real(np.sum(self.A0[ifo][:-1]*np.conj(r0[:-1]) + self.A1[ifo][:-1]*np.conj(r1[:-1])))
            h_inner_h += np.real(np.sum((self.B0[ifo][:-1]*np.abs(r0[:-1])**2) + 2*self.B1[ifo][:-1]*np.real(r0[:-1]*np.conj(r1[:-1]))))
        
        log_l_ratio = d_inner_h - h_inner_h/2.

        return log_l_ratio

    def log_likelihood(self):
        """
        Function putting everything together to get the log likelihood
        of the event
        """

        return self.log_likelihood_ratio() + self.noise_log_l


def get_binned_td_wf_polarizations(parameters, waveform_generator, idxs):
    """
    Function to get the poalrizations for relative binning
    when the waveform used is a time domain waveform

    ARGS:
    -----
    - parameters: the parameters for which we want to 
                  compute the polarizations
    - waveform_generator: the bilby waveform generator
                          object to use for the WF generation
    - idxs: the idxs at which we have frequencies for the
            relative binning scheme

    RETURNS:
    -------
    - the array of values for the + and x polarizations
      at the values of the bins 
    """

    if 'lambda_1' not in parameters.keys() and 'lambda_tilde' not in parameters.keys(): #BBH case
        params, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters)
        params['lambda_1'] = 0.
        params['lambda_2'] = 0.
    else:
        params, _ = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(parameters)

    # convert parameters appropriately for lal
    lum_dist = params['luminosity_distance'] * 1e6 * bilby.core.utils.parsec
    m1 = params['mass_1'] * bilby.core.utils.solar_mass
    m2 = params['mass_2'] * bilby.core.utils.solar_mass

    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(theta_jn = params['theta_jn'],
                                                                                          phi_jl = params['phi_jl'],
                                                                                          tilt_1 = params['tilt_1'],
                                                                                          tilt_2 = params['tilt_2'],
                                                                                          phi_12 = params['phi_12'],
                                                                                          a_1 = params['a_1'],
                                                                                          a_2 = params['a_2'],
                                                                                          mass_1 = m1, mass_2 = m2,
                                                                                          reference_frequency = waveform_generator.waveform_arguments['reference_frequency'],
                                                                                          phase = params['phase'])
    # make info for lal format
    lal_pars = lal.CreateDict()

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(lal_pars,
                                                      int(waveform_generator.waveform_arguments['pn_spin_order']))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(lal_pars,
                                                       int(waveform_generator.waveform_arguments['pn_tidal_order']))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(lal_pars,
                                                       int(waveform_generator.waveform_arguments['pn_phase_order']))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lal_pars,
                                                           int(waveform_generator.waveform_arguments['pn_amplitude_order']))
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, 
                                                       params['lambda_1'])
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, 
                                                       params['lambda_2'])

    # use the fourier transform function from lal
    hpl, hcr = bilby.gw.utils.lalsim_SimInspiralFD(m1, m2, s1x, s1y, s1z,
                                                   s2x, s2y, s2z, lum_dist,
                                                   iota, params['phase'],
                                                   0., 0., 0.,
                                                   1./waveform_generator.duration,
                                                   waveform_generator.waveform_arguments['minimum_frequency'],
                                                   waveform_generator.waveform_arguments['maximum_frequency'],
                                                   waveform_generator.waveform_arguments['reference_frequency'],
                                                   lal_pars,
                                                   waveform_generator.waveform_arguments['waveform_approximant'])
    binned_polas = {'plus' : hpl.data.data[idxs],
                    'cross' : hcr.data.data[idxs]}

    return binned_polas

def get_binned_td_wf_detector_response(ifos, waveform_polarizations, parameters, idxs):
    """
    Function computing the detector reponsed for the 
    relative binning when the waveform is a time domain
    waveform 

    ARGS:
    -----
    - ifos: bilby interferometer list. The interferometes for which 
            the detector response should be computed
    - waveform_polarization: dict of arrays. The waveform polarizations 
                             under consideration
    - parameters: dict. The parameters under consideration. 
    - idxs: the indices at which are the frequencies 
            for the relative binning scheme

    RETURNS:
    --------
    - signals: dict. The signal observed in the detectors for the 
               specified polarization-parameters combination
    """

    signals = {}

    for ifo in ifos:
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = ifo.antenna_response(parameters['ra'], parameters['dec'],
                                                parameters['geocent_time'],
                                                parameters['psi'], mode)
            
            signal[mode] = waveform_polarizations[mode]*det_response

        signal_ifo = sum(signal.values())
        # compute and apply the time shift 
        time_shift = ifo.time_delay_from_geocenter(parameters['ra'],
                                                   parameters['dec'],
                                                   parameters['geocent_time'])
        dt_geocent = parameters['geocent_time'] - ifo.strain_data.start_time
        dt = dt_geocent + time_shift

        # apply the time shift
        signal_ifo = signal_ifo * np.exp(-1j*2*np.pi*dt*ifo.strain_data.frequency_array[idxs])

        # do calibration if needed
        signal_ifo *= ifo.calibration_model.get_calibration_factor(ifo.strain_data.frequency_array[idxs],
                                                                   prefix = 'recalib_{}_'.format(ifo.name),
                                                                   **parameters)
        signals[ifo.name] = signal_ifo

    return signals

class TimeDomainRelativeBinning(Likelihood):
    """
    Class implementing the relative binning for a time
    domain waveform
    """

    def __init__(self, interferometers, waveform_generator, model, priors, delta = 0.03):
        """
        Initialization of the class. 

        ARGS:
        -----
        - interferometers: list of interferometers that contain the signal
        - waveform_generator: the bilby waveform generator object that should
                              be used to setup the run
        - model: dictionary with the model parameters to be used
                 for the analysis
        - priors: the priors used for the different parameters
        - delta: the precision factor to be used in the run. Default is 0.03
        """

        super(TimeDomainRelativeBinning, self).__init__(dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.priors = priors
        self.model = model

        # retrieve some useful information for the rest of the run 
        if 'minimum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_min = self.waveform_generator.waveform_arguments['minimum_frequency']
        else:
            self.f_min = 20.
        if 'maximum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_max = self.waveform_generator.waveform_arguments['maximum_frequency']
        else:
            self.f_max = self.waveform_generator.sampling_frequency/2. # take nyquist
        if 'pn_spin_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_spin_order'] = -1
        if 'pn_phase_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_phase_order'] = -1
        if 'pn_tidal_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_tidal_order'] = -1
        if 'pn_amplitude_order' not in self.waveform_generator.waveform_arguments.keys():
            self.waveform_generator.waveform_arguments['pn_amplitude_order'] = 0.

        self.duration = self.waveform_generator.duration

        self.make_relative_binning_information(delta = delta)

        # dictionary to be filled by the sampler
        self.parameters = {}

        # additional speed up
        self.noise_log_l = self.noise_log_likelihood()

    @property
    def priors(self):
        return self._prior
    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        else:
            self._prior = None

    def make_relative_binning_information(self, delta = 0.03):
        """
        Function computing the summary data for the given model

        ARGS:
        -----
        - delta: precision required when computing the summary data
        """

        # make the model data 
        model_polas = self.waveform_generator.frequency_domain_strain(parameters = self.model)
        models = dict()
        for ifo in self.interferometers:
            models[ifo.name] = ifo.get_detector_response(model_polas, self.model)

        # make the array of frequencie and compute the correct values
        # for the bin edges
        f_nt = np.linspace(self.f_min, self.f_max, int(1e4))
        ga = np.array([-5./3, -2./3, 1., 5./3, 7./3])

        dalpha = 2*np.pi/np.abs(self.f_min**ga - self.f_max**ga)
        dphi = np.sum([np.sign(g)*d*f_nt**g for d,g in zip(dalpha, ga)], axis = 0)
        dphi -= dphi[0]

        # construct the frequency bins
        nbin = int(np.ceil(dphi[-1]/delta))
        dphi_grid = np.linspace(dphi[0], dphi[-1], nbin+1)
        self.fbin = np.interp(dphi_grid, dphi, f_nt)

        # set the bin edges and widths adapted to the frequencies
        # for which we have data points
        self.fbin_idxs = np.unique(np.argmin(np.abs(self.waveform_generator.frequency_array[:, np.newaxis] - self.fbin), axis = 0))
        self.fbin = self.waveform_generator.frequency_array[self.fbin_idxs]
        self.Nbin = len(self.fbin) - 1
        self.fm = (self.fbin[1:] + self.fbin[:-1])/2.
        self.binwidths = self.fbin[1:] - self.fbin[:-1]

        # compute the actual summary data
        self.A0, self.A1, self.B0, self.B1 = dict(), dict(), dict(), dict()
        for ifo in self.interferometers:
            d = ifo.strain_data.frequency_domain_strain.copy()
            h = models[ifo.name].copy()
            psd = ifo.power_spectral_density_array.copy()
            fs = ifo.strain_data.frequency_array

            self.A0[ifo.name] = (4./self.duration)*np.array([np.sum(d[self.fbin_idxs[b]:self.fbin_idxs[b+1]]*\
                                np.conj(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])/psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])
                                for b in range(self.Nbin)])
            self.A1[ifo.name] = (4./self.duration)*np.array([np.sum((d[self.fbin_idxs[b]:self.fbin_idxs[b+1]]*\
                                 np.conj(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])/psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])*\
                                 (fs[self.fbin_idxs[b]:self.fbin_idxs[b+1]] - self.fm[b]))
                                 for b in range(self.Nbin)])
            self.B0[ifo.name] = (4./self.duration)*np.array([np.sum((np.abs(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])**2)/\
                                 psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])
                                 for b in range(self.Nbin)])
            self.B1[ifo.name] = (4./self.duration)*np.array([np.sum(((np.abs(h[self.fbin_idxs[b]:self.fbin_idxs[b+1]])**2)/\
                                 psd[self.fbin_idxs[b]:self.fbin_idxs[b+1]])*(fs[self.fbin_idxs[b]:self.fbin_idxs[b+1]] - self.fm[b]))
                                 for b in range(self.Nbin)])

        # keep the waveform model only for the correct points
        self.h0 = dict()
        self.idxs_zeros = dict()
        for ifo in models.keys():
            self.h0[ifo] = models[ifo][self.fbin_idxs]
            # keep indices where model is zero to avoid
            # numerical errors later on
            self.idxs_zeros[ifo] = np.where(self.h0[ifo] == 0)[0]


    def noise_log_likelihood(self):
        """
        Function computing the noise log likelihood
        """
        log_l = 0.

        for ifo in self.interferometers:
            mask =ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))

    def log_likelihood_ratio(self):
        """
        Function computing the likelihood ratio 
        using the relative binning method and the summary
        data computed above
        """

        polas = get_binned_td_wf_polarizations(self.parameters, self.waveform_generator, self.fbin_idxs)
        h = get_binned_td_wf_detector_response(self.interferometers, polas, self.parameters,
                                               self.fbin_idxs)

        d_inner_h = 0.
        h_inner_h = 0.

        for ifo in h.keys():
            r = h[ifo]/self.h0[ifo]
            r[self.idxs_zeros[ifo]] = 0.

            rpl = r[1:]
            rmin = r[:-1]

            r0 = (rpl + rmin)/2.
            r1 = (rpl - rmin)/self.binwidths

            d_inner_h += np.real(np.sum(self.A0[ifo][:-1]*np.conj(r0[:-1]) + self.A1[ifo][:-1]*np.conj(r1[:-1])))
            h_inner_h += np.real(np.sum((self.B0[ifo][:-1]*np.abs(r0[:-1])**2) + 2*self.B1[ifo][:-1]*np.real(r0[:-1]*np.conj(r1[:-1]))))
        
        log_l_ratio = d_inner_h - h_inner_h/2.

        return log_l_ratio

    def log_likelihood(self):
        """
        Function putting everything above together as to 
        have the log likelihood ratio for the samples under
        consideration
        """

        return self.log_likelihood_ratio() + self.noise_log_l
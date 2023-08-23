import numpy as np
from tqdm import tqdm
import bilby
from bilby.core.likelihood import Likelihood
from numba import njit
import lal
from . import  xphm
from bilby.gw.utils import noise_weighted_inner_product
import matplotlib.pyplot as plt
from bilby.gw.source import lal_binary_black_hole
from . import separate_l_likelihood





def lensed_lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, n_phase, **kwargs):
    """

    function to add morse phase in the bbh waveform 


    """
    polas = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    #print(kwargs)   
    for pola in ['plus', 'cross']:
        #print('shifting phase by', n_phase)
        polas[pola] = np.exp(-1j*n_phase)*polas[pola]
    return polas



def make_lensed_injection(injection_parameters, image_index):
    common_parameters = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'phase',
                    'ra', 'dec', 'theta_jn', 'psi']





    lens_injection = {}

    for p in common_parameters:
        lens_injection[p] = injection_parameters[p]


    if image_index==1:
        lens_injection['geocent_time'] = injection_parameters['geocent_time']
        lens_injection['n_phase'] = injection_parameters['n_phase_1']
        lens_injection['luminosity_distance'] = injection_parameters['luminosity_distance']


        return lens_injection

    else:


        lens_injection['geocent_time'] = injection_parameters['geocent_time']+injection_parameters[f'delta_t_{image_index}1']
        lens_injection['luminosity_distance'] = injection_parameters['luminosity_distance']*np.sqrt(injection_parameters[f'mu_{image_index}1'])
        lens_injection['n_phase'] = injection_parameters['n_phase_1']+injection_parameters[f'delta_n_{image_index}1']

        return lens_injection
    


def lal_f_max(parameters, waveform_arguments, reference_frequency=50.):


    parameters = relby_jointpe_conversion(parameters, waveform_arguments)
    
    spin1 = np.sqrt(parameters['chi_1x']**2+parameters['chi_1y']**2+parameters['chi_1z']**2)
    spin2 = np.sqrt(parameters['chi_2x']**2+parameters['chi_2y']**2+parameters['chi_2z']**2)
    mtotal = parameters['mass_1']+parameters['mass_2']

    chi_eff = (parameters['mass_1']*spin1+parameters['mass_2']*spin2)/mtotal


    msun_sec = 4.925491025543575903411922162094833998e-6
    #print(chi_eff, 'chi effectice')
    if chi_eff>0.99:
        return 0.33/(mtotal*msun_sec)
    else:
        return 0.3/(mtotal*msun_sec)



def relby_jointpe_conversion(parameters, waveform_arguments):

    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'],
                                                                                        parameters['mass_ratio'])

    iota, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(parameters['theta_jn'], parameters['phi_jl'], parameters['tilt_1'], parameters['tilt_2'],
                                                                                                                        parameters['phi_12'], parameters[
                                                                                                                            'a_1'], parameters['a_2'],
                                                                                                                        parameters['mass_1'] * lal.MSUN_SI, parameters['mass_2'] * lal.MSUN_SI,
                                                                                                                                                    waveform_arguments['reference_frequency'], parameters['phase'])

    relby_parameters = {}

    for k in parameters.keys():
        if k not in ['phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'a_1', 'a_2', 'theta_jn']:
            relby_parameters[k] = parameters[k]

    relby_parameters['mass_1'] = parameters['mass_1']
    relby_parameters['mass_2'] = parameters['mass_2']

    relby_parameters['inclination'] = iota
    relby_parameters['chi_1x'] = chi_1x
    relby_parameters['chi_1y'] = chi_1y
    relby_parameters['chi_1z'] = chi_1z
    relby_parameters['chi_2x'] = chi_2x
    relby_parameters['chi_2y'] = chi_2y
    relby_parameters['chi_2z'] = chi_2z

    '''
    {'mass_1': parameters['mass_1'], 'mass_2': parameters['mass_2'],
        'chirp_mass': parameters['chirp_mass'], 'mass_ratio': parameters['mass_ratio'],

        'chi_1x': chi_1x,
        'chi_1y': chi_1y,
        'chi_1z': chi_1z,
        'chi_2x': chi_2x,
        'chi_2y': chi_2y,
        'chi_2z': chi_2z,
        'luminosity_distance': parameters['luminosity_distance'],
        'ra': parameters['ra'],
        'dec': parameters['dec'],
        'psi': parameters['psi'],
        'inclination': iota,
        'phase': parameters['phase'],
        'geocent_time': parameters['geocent_time'],
        'n_phase_1': parameters['n_phase_1'],
        'delta_t': parameters['delta_t'],
        'delta_n': parameters['delta_n'],
        'relative_magnification': parameters['relative_magnification']}

    '''

    relby_parameters['theta_jn'] = xphm.theta_jn_func_op(relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'],
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'],
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'], waveform_arguments['reference_frequency'])

    relby_parameters['kappa'] = xphm.kappa_func_op(relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'],
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'],
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'], waveform_arguments['reference_frequency'])

    return relby_parameters


@njit
def jit_interp(fs, rebinned_c, Ctotal, num_modes, num_ifos, Creturn):
    '''
    this function works faster without jit, or as fast probably
    '''
    for i in range(num_ifos):
        for j in range(num_modes):
            Creturn[i, j, :] = np.interp(fs, rebinned_c, Ctotal[i, j, :])
    return Creturn


def inn_prod(x, y, psd, T):
    """
    Computes the inner product between two frequency series.
    x, y: frequency series arrays
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    """
    return np.real(np.sum(4*np.conj(x)*y/psd/T))


def get_lensed_binned_detector_response(ifos_dict, parameters, fs, waveform_arguments, mode_array):

    #parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
     #   parameters['chirp_mass'], parameters['mass_ratio'])

    
    Cplus, Ccross, _hL = xphm.compute_C_prefactors(
            parameters, fs, waveform_arguments, np.array(mode_array), fs)

    Fplus = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
        parameters['geocent_time'], parameters['psi'], 'plus') for ifo in ifos_dict['image_1']])
    Fcross = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
        parameters['geocent_time'], parameters['psi'], 'cross') for ifo in ifos_dict['image_1']])

    dt_geocent = np.array([ifo.time_delay_from_geocenter(
        parameters['ra'], parameters['dec'], parameters['geocent_time']) for ifo in ifos_dict['image_1']])
    # this is basically adding time delay from geocent and duration of the signal
    dt_total = dt_geocent+(parameters['geocent_time']- ifos_dict['image_1'][0].start_time)

    Fplus = Fplus[:, np.newaxis, np.newaxis]
    Fcross = Fcross[:, np.newaxis, np.newaxis]

    Cplus_projected = Fplus*Cplus
    Ccross_projected = Fcross*Ccross

    Ctotal = Cplus_projected+Ccross_projected

    exp_timeshift = np.exp(-1j*2*np.pi*fs*dt_total[:, np.newaxis])
    if _hL is not None:
        hL = exp_timeshift[:, np.newaxis, :] * \
            np.exp(-1j*parameters['n_phase_1'])*_hL[np.newaxis, :, :]
    else:
        hL = None

    ######################## calculate same quantities for the lensed event ###################################
    ############################################################################################################

    Fplus_lensed = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
        parameters['geocent_time']+parameters['delta_t_21'], parameters['psi'], 'plus') for ifo in ifos_dict['image_2']])
    Fcross_lensed = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
        parameters['geocent_time']+parameters['delta_t_21'], parameters['psi'], 'cross') for ifo in ifos_dict['image_2']])

    dt_geocent_lensed = np.array([ifo.time_delay_from_geocenter(parameters['ra'], parameters['dec'], parameters['geocent_time']+parameters['delta_t_21']) for ifo in ifos_dict['image_2']])
    dt_total_lensed = dt_geocent_lensed + \
        ((parameters['geocent_time']+parameters['delta_t_21'])-ifos_dict['image_2'][0].start_time)

    Fplus_lensed = Fplus_lensed[:, np.newaxis, np.newaxis]
    Fcross_lensed = Fcross_lensed[:, np.newaxis, np.newaxis]

    Cplus_projected_lensed = Fplus_lensed*Cplus
    Ccross_projected_lensed = Fcross_lensed*Ccross

    Ctotal_lensed = Cplus_projected_lensed+Ccross_projected_lensed

    exp_timeshift_lensed = np.exp(-1j*2*np.pi*fs*dt_total_lensed[:, np.newaxis])
    if _hL is not None:
        hL_lensed = exp_timeshift_lensed[:, np.newaxis, :]*_hL[np.newaxis, :, :]
        n_phase_2 = parameters['n_phase_1']+parameters['delta_n_21']
        hL_lensed = np.exp(-1j*n_phase_2)*hL_lensed/np.sqrt(parameters['mu_21'])
    else:
        hL_lensed = None

    ############################################################################################################
    ############################################################################################################

    #print(np.shape(hL_lensed), np.shape(Ctotal_lensed))

    return {'image_1':{'hL':hL, 'C': Ctotal}, 'image_2':{'hL': hL_lensed, 'C': Ctotal_lensed}}


def get_lensed_quad_binned_detector_response(ifos_dict, parameters, fs, waveform_arguments, mode_array, rebinned_c=None):

    '''
    This function returns dictionary for the hL and C projectd onto the detectors for four images

    '''



    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        parameters['chirp_mass'], parameters['mass_ratio'])


    #### waveform is called just once #######

    if rebinned_c is not None:
        Cplus, Ccross, _hL = xphm.compute_C_prefactors(
            parameters, rebinned_c, waveform_arguments, np.array(mode_array), fs)

    else:
        Cplus, Ccross, _hL = xphm.compute_C_prefactors(
            parameters, fs, waveform_arguments, np.array(mode_array), fs)


    ### End of waveform call ##################

    parameters['delta_t_11'] = 0. ## time delay between image 1 and 1. I need this param to write a for loop
    parameters['delta_n_11'] = 0. ## same quantity for morse shift
    parameters['mu_11'] = 1. ## same quantity for magnification
    

    ## in the following for loop, I populate this dictionary with plus and cross wf projected on the detectors
    plus_cross_dict = {}

    for jj in [1, 2, 3, 4]:
        
        ######## calculate antenna response fuction for each image ##################

        Fplus = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
                parameters['geocent_time']+parameters[f'delta_t_{jj}1'], parameters['psi'], 'plus') for ifo in ifos_dict[f'ifos_image_{jj}']])
        Fplus = Fplus[:, np.newaxis, np.newaxis]

        Fcross = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'],
                                    parameters['geocent_time']+parameters[f'delta_t_{jj}1'], parameters['psi'], 'cross') for ifo in ifos_dict[f'ifos_image_{jj}']])
        Fcross = Fcross[:, np.newaxis, np.newaxis]


        Cprojected = (Fplus*Cplus) + (Fcross*Ccross)

        plus_cross_dict[f'image_{jj}/Cprefactors'] = Cprojected


        #############################################################################
        #############################################################################

        ############ calculate time shift for each image ###########################


        ts_a = np.array([ifo.time_delay_from_geocenter(
        parameters['ra'], parameters['dec'], parameters['geocent_time']+parameters[f'delta_t_{jj}1']) for ifo in ifos_dict[f'ifos_image_{jj}']])

        ts_b = parameters['geocent_time']+parameters[f'delta_t_{jj}1'] - ifos_dict[f'ifos_image_{jj}'][0].start_time

        ts_total = ts_a + ts_b

        exp_timeshift  = np.exp(-1j*2*np.pi*fs*ts_total[:, np.newaxis])

        if _hL is not None:
            morse_shift = parameters['n_phase_1']+parameters[f'delta_n_{jj}1']  ## no pi needed, 
            hL = exp_timeshift[:, np.newaxis, :]*np.exp(-1j*morse_shift)*_hL[np.newaxis, :, :]
            plus_cross_dict[f'image_{jj}/hL'] = hL/np.sqrt(parameters[f'mu_{jj}1'])

        else:
            plus_cross_dict[f'image_{jj}/hL'] = None



        #############################################################################
        #############################################################################



    return plus_cross_dict


class RealDataJointLikelihood(Likelihood):
    def __init__(self, ifos_dict, fiducial_model, test_model, priors, mode_array, waveform_arguments, total_err):
        super(RealDataJointLikelihood, self).__init__(dict())
        
        self.ifos_dict = ifos_dict

        self.fiducial_model = {} #fiducial_model  ## a dictionary 
        self.test_model = {}#test_model        ## a dictionary
        for key in fiducial_model.keys():
            self.fiducial_model[key] = relby_jointpe_conversion(fiducial_model[key], waveform_arguments)
            self.test_model[key] = relby_jointpe_conversion(test_model[key], waveform_arguments)

        self.priors = priors
        self.mode_array = mode_array
        self.waveform_arguments = waveform_arguments



        self.total_err = total_err

        self.duration = self.ifos_dict['image_1'].duration ### both signal must have same duration
        self.N_initial = 200 ## hard coding here ###

        self.fs = {}
        self.fiducial_waveform = {}
        self.test_waveform = {}
        self.data = {}
        self.psd = {}
        self.mask = {}

        print('Populating the data, psd, fiducial waveform, test waveform and frequency grids for image 1 and image 2')
        for i in [1, 2]:

            minimum_frequency = self.ifos_dict[f'image_{i}'][0].minimum_frequency
            maximum_frequency = self.ifos_dict[f'image_{i}'][0].maximum_frequency


            self.mask = self.ifos_dict[f'image_{i}'][0].frequency_mask ### if this fails shift add an index 0

            self.fs[f'image_{i}'] = self.ifos_dict[f'image_{i}'].frequency_array[self.mask]

            self.fiducial_waveform[f'image_{i}'] = {}
            self.fiducial_waveform[f'image_{i}']['h0'], self.fiducial_waveform[f'image_{i}']['C'] = separate_l_likelihood.get_binned_detector_response(self.ifos_dict[f'image_{i}'], self.fiducial_model[f'image_{i}'], self.fs[f'image_{i}'], self.waveform_arguments, self.mode_array)


            self.test_waveform[f'image_{i}'] = {}
            self.test_waveform[f'image_{i}']['h0'], self.test_waveform[f'image_{i}']['C']  = separate_l_likelihood.get_binned_detector_response(self.ifos_dict[f'image_{i}'], self.test_model[f'image_{i}'], self.fs[f'image_{i}'], self.waveform_arguments, self.mode_array)

            self.data[f'image_{i}'] = np.array([ifo.frequency_domain_strain[self.mask] for ifo in ifos_dict[f'image_{i}']])
            self.psd[f'image_{i}'] = np.array([ifo.power_spectral_density_array[self.mask] for ifo in ifos_dict[f'image_{i}']])



        self.noise_log_l = self.noise_log_likelihood()
        self.master_fbin_grid, self.master_fbin_ind, self.master_fm, self.h_fbin, self.summary_data = self.setup_bins()
        print('Total f bins used by relative binning grid', len(self.master_fbin_grid))
        print('Total f bins used by regular PE grid for image 1 and image 2', len(self.fs['image_1']), len(self.fs['image_2']))


        self.fiducial_ll = self.fiducial_log_likelihood()
        print('Full fiducial likelihood:', self.fiducial_ll)
        #exit()

        self.parameters = {}




    def noise_log_likelihood(self):
        log_l = 0.

        for i in [1, 2]:
            for ifo in self.ifos_dict[f'image_{i}']:
                mask = ifo.frequency_mask
                log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask], ifo.frequency_domain_strain[mask], 
                    ifo.power_spectral_density_array[mask], self.duration)/2.


        return float(np.real(log_l))


    def log_likelihood_ratio(self):
        p = relby_jointpe_conversion(self.parameters, self.waveform_arguments)
        plus_cross_dict = get_lensed_binned_detector_response(self.ifos_dict['image_1'], self.ifos_dict['image_2'], p, self.master_fbin_grid, self.waveform_arguments, mode_array = self.mode_array)

        if plus_cross_dict['image_1']['h0'] is not None:

            log_likelihood_ratio = 0.

            for i in [1, 2]:
                r = (plus_cross_dict[f'image_{i}']['h0'])/(self.h_fbin[f'image_{i}']['h0'])
                s = (plus_cross_dict[f'image_{i}']['C'])/(self.h_fbin[f'image_{i}']['C'])

                r0, r1 = self.compute_bin_coefficients(self.master_fbin_grid, r)
                s0, s1 = self.compute_bin_coefficients(self.master_fbin_grid, s)

                Zdh, Zhh = self.compute_overlaps(r0, r1, s0, s1, 
                    self.summary_data[f'image_{i}']['A0'], self.summary_data[f'image_{i}']['A1'], self.summary_data[f'image_{i}']['B0'], self.summary_data[f'image_{i}']['B1'])

                log_likelihood_ratio += (Zdh-0.5*Zhh)

            return log_likelihood_ratio

        else:
            print('Cannot generate the waveform at ', p)
            print('Setting the likelihood to -np.inf')
            return -np.inf



    def log_likelihood(self):
        return self.log_likelihood_ratio()+self.noise_log_l


    def fiducial_log_likelihood(self):
        store_fiducial = 0
        for i in [1, 2]:
            r = (self.h_fbin[f'image_{i}']['h0'])/(self.h_fbin[f'image_{i}']['h0'])
            s = (self.h_fbin[f'image_{i}']['C'])/(self.h_fbin[f'image_{i}']['C'])   

            r0, r1 = self.compute_bin_coefficients(self.master_fbin_grid, r)
            s0, s1 = self.compute_bin_coefficients(self.master_fbin_grid, s)
            Zdh, Zhh = self.compute_overlaps(r0, r1, s0, s1, 
                    self.summary_data[f'image_{i}']['A0'], self.summary_data[f'image_{i}']['A1'], self.summary_data[f'image_{i}']['B0'], self.summary_data[f'image_{i}']['B1'])

            store_fiducial += (Zdh-0.5*Zhh)
            print(f'Image: {i} Zdh: {Zdh} Zhh: {Zhh}')

        return store_fiducial




    def compute_summary_data(self, fbin, Nbin, fbin_ind, fm, f, strain, h0, C, psd, ndet):
    
        
        Nmode = 5#signal.shape[1]  # this is hard coded

        T = self.duration


        A0 = 4./T*np.array([[[np.sum(strain[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*C[i, l, fbin_ind[b]:fbin_ind[b+1]]) /
                           psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        A1 = 4./T*np.array([[[np.sum(strain[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*C[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        B0 = 4./T*np.array([[[[np.sum(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*C[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, ll, fbin_ind[b]:fbin_ind[b+1]]*C[i, ll, fbin_ind[b]:fbin_ind[b+1]]) /
                           psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
        B1 = 4./T*np.array([[[[np.sum(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*C[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, ll, fbin_ind[b]:fbin_ind[b+1]]*C[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]
                           * (f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])

        return {'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1}




    def setup_bins(self):

        '''
        This function returns the relative binning grid, waveform on that grid, and summary data for image 1 and image 2

        Which are then made global variables

        '''
        master_fbin_grid = np.array([])
        master_fbin_ind = np.array([])

        for k in [1, 2]:
            fbin, fbin_ind, _ = self.get_binning_data(self.N_initial, image_index = k)

            print(f'relative binning grid length of image number {k}:', len(fbin))
            master_fbin_grid = np.union1d(fbin, master_fbin_grid)
            master_fbin_ind = np.union1d(fbin_ind, master_fbin_ind)

        master_fbin_ind = np.asarray(master_fbin_ind, dtype = int)


        master_fm = (master_fbin_grid[1:] + master_fbin_grid[:-1])/2.

        Nbin = len(master_fbin_ind)-1

        summary_data = {}
        h_fbin = {}

        for i in [1, 2]:
            ndet = len(self.ifos_dict[f'image_{i}'])
            summary_data[f'image_{i}'] = self.compute_summary_data(master_fbin_grid, Nbin, master_fbin_ind, master_fm, self.fs[f'image_{i}'], self.data[f'image_{i}'], self.fiducial_waveform[f'image_{i}']['h0'], 
                                self.fiducial_waveform[f'image_{i}']['C'], self.psd[f'image_{i}'], ndet = ndet)

            h_fbin[f'image_{i}'] = {}
            h_fbin[f'image_{i}']['h0'] = (self.fiducial_waveform[f'image_{i}']['h0'])[ :, :, master_fbin_ind]
            h_fbin[f'image_{i}']['C'] = (self.fiducial_waveform[f'image_{i}']['C'])[:, :, master_fbin_ind]


        return master_fbin_grid, master_fbin_ind, master_fm, h_fbin, summary_data



    def get_binning_data(self, N0, image_index, i = 0):


        print('bin progression', 'iteration', i , 'input bins', N0)

    
        fbin, Nbin, fbin_ind, fm = self.get_step_bin_grid(N0, image_index)

        print('output bins', Nbin, '###########\n')

        if Nbin==N0:
            return fbin, fbin_ind, fm

        elif i==8:
            print('Reached max of iteration:',i)
            print('Breaking and returning the grid')
            return fbin, fbin_ind, fm

        else:
            return self.get_binning_data(Nbin, image_index, i = i+1)


    def compute_bin_error(self, f_lo, f_hi, image_index):
        """
        
        h0: fiducial model modes, indices: detector, mode, frequency
        h: test model modes, indices: detector, mode, frequency
        f_lo: target lower frequency for the bin, this exact frequency may not be in the frequency array
        f_hi: target upper frequency for the bin, this exact frequency may not be in the frequency array
        sign: boolean, if True: returns the error (relative binning - exact) with its sign, this option is used for investigating error in relative binning
                       if False: returns the absolute value of the error, this option is used in the bisecting bin selection algorithm
        
        Cfbin: set to None when simultaneously in hL and C. Mention the Value when rebinning in C
        """
        # build length 2 array for small bin

        ii = image_index
        ndet = len(self.ifos_dict[f'image_{ii}'])
        fbin = np.array([f_lo, f_hi])
        fbin_ind = np.unique(np.argmin(np.abs(self.fs[f'image_{image_index}'][:, np.newaxis] - fbin), axis=0))
        fbin = self.fs[f'image_{image_index}'][fbin_ind]
        Nbin = 1
        fm = 0.5*(fbin[1:]+fbin[:-1])

        summary_data = self.compute_summary_data(fbin, Nbin, fbin_ind, fm, self.fs[f'image_{image_index}'], self.data[f'image_{ii}'], self.fiducial_waveform[f'image_{ii}']['h0'], self.fiducial_waveform[f'image_{ii}']['C'], self.psd[f'image_{ii}'], ndet)
        r = (self.test_waveform[f'image_{ii}']['h0'])/(self.fiducial_waveform[f'image_{ii}']['h0'])
        r = r[:,:,fbin_ind]

        s = (self.test_waveform[f'image_{ii}']['C'])/(self.fiducial_waveform[f'image_{ii}']['C'])
        s = s[:,:,fbin_ind]

        t_allmodes = np.sum((self.test_waveform[f'image_{ii}']['C'])*(self.test_waveform[f'image_{ii}']['h0']), axis = 1)

        exact_lnL = np.sum([inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.data[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.psd[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.duration) - 1/2*inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.psd[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.duration) for i in range(ndet)])



        r0, r1 = self.compute_bin_coefficients(fbin, r)
        C0, C1 = self.compute_bin_coefficients(fbin, s)

        Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, summary_data['A0'], summary_data['A1'], summary_data['B0'], summary_data['B1'])

        rb_lnL = Zdh - 1/2*Zhh


        return np.abs(rb_lnL - exact_lnL)



    def compute_overlaps(self, r0, r1, C0, C1, A0, A1, B0, B1):
        """
        Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf) used in advanced mode by mode relative binning (Scheme 1).
        r0, r1: linear coefficients for the ratio of time-dependent L-frame modes over fiducial time-dependent L-frame modes, indices: detector, mode, frequency
        C0, C1: linear coefficients for the coefficient of the time-dependent L-frame modes, indices: detector, mode, frequency
        A0, A1, B0, B1: summary data for mode-by mode relative binning, A indices: detector, mode, frequency, B indices: detector, mode, mode, frequency
        """
        Zdh = np.real(np.sum( A0*np.conj(r0)*np.conj(C0) + A1*( (np.conj(r0)*np.conj(C1)) + (np.conj(r1)*np.conj(C0)) ) ))

        Zhh = np.real(np.sum( B0*r0[:, np.newaxis, :, :]*C0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*np.conj(C0[:, :, np.newaxis, :])
                            + B1*( r0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*(np.conj(C0[:, :, np.newaxis, :])*C1[:, np.newaxis, :, :] + C0[:, np.newaxis, :, :]*np.conj(C1[:, :, np.newaxis, :])) + C0[:, np.newaxis, :, :]*np.conj(C0[:, :, np.newaxis, :])*(np.conj(r0[:, :, np.newaxis, :])*r1[:, np.newaxis, :, :] + r0[:, np.newaxis, :, :]*np.conj(r1[:, :, np.newaxis, :])) ) ))

        return Zdh, Zhh


    def compute_bin_coefficients(self, fbin, r):
        """
        Computes the bin coefficients used in mode-by-mode relative binning.
        fbin: array of frequencies of the bin edges (Hz)
        r: array of ratios of model over fiducial model, indices: detector, mode, frequency
        """
        binwidths = fbin[1:] - fbin[:-1]

        # rplus: right edges of bins
        rplus = r[:, :, 1:]
        # rminus: left edges of bins
        rminus = r[:, :, :-1]

        r0 = 0.5 * (rplus + rminus)
        r1 = (rplus - rminus) / binwidths[np.newaxis, np.newaxis, :]

        return r0, r1



    def get_step_bin_grid(self, N, image_index):

        #if quad_sum:
        #    print('quad sum')
        Emax = self.total_err/np.sqrt(N)
        #else:
        #    print('direct sum')
        #    Emax = self.total_err/N
        #print('Emax of the loop', Emax)

        output_step_bins = [self.fs[f'image_{image_index}'][0]]
        output_step_bins_index = [0]
        i = 0
        while i <len(self.fs[f'image_{image_index}']):
            #print('step flow bin index', i)
            j = 1 # steps in which the bins will increase
            while (i+j)<len(self.fs[f'image_{image_index}']):
                #print('step adding bin', j)
                flo = self.fs[f'image_{image_index}'][i]
                fhi = self.fs[f'image_{image_index}'][i+j]
                add_bins = i+j
                #print('flo and fhi', flo, fhi)
                bin_error = self.compute_bin_error(flo, fhi, image_index)
                j+=1

                if bin_error>Emax:
                    i = i+j-1
                    #print('we reached', i+j-1)
                    output_step_bins.append(self.fs[f'image_{image_index}'][add_bins])
                    output_step_bins_index.append(add_bins)

                    #print('number of bins combined', j)
                    #print('moving to the next bin \n\n')
                    break
            if (i+j)==len(self.fs[f'image_{image_index}']):
                print('Breaking because we reached the end')
                break
        #print('output bins', output_step_bins)
        output_step_bins = np.unique(np.array(output_step_bins))
        Nbin = len(output_step_bins)
        output_step_bins_index = np.unique(np.array(output_step_bins_index))
        fm = (output_step_bins[1:] + output_step_bins[:-1])/2
        return np.unique(np.array(output_step_bins)), len(output_step_bins)-1, np.unique(np.array(output_step_bins_index)), fm














class JointPELikelihood(Likelihood):
    def __init__(self, ifos_dict, fiducial_model, test_model, priors, mode_array, waveform_arguments, total_err, quad_sum = True, grid_choice = 'step'):

        super(JointPELikelihood, self).__init__(dict())

        self.ifos_dict = ifos_dict

        self.fiducial_model = fiducial_model
        self.test_model = test_model
        
        self.priors = priors
        self.mode_array = mode_array
        self.waveform_arguments = waveform_arguments
        self.total_err = total_err
        self.duration = self.ifos_dict['image_1'].duration ### both signals have same duration
        self.N_initial = 200 #.01*len(self.fs)  # what is the purpose of this?
        self.fid_fmax = np.min([lal_f_max(self.fiducial_model['image_1'], self.waveform_arguments), self.ifos_dict['image_1'][0].frequency_array[-1]])
        


        self.minimum_frequency = self.ifos_dict['image_1'][0].minimum_frequency

        self.mask = np.where((self.ifos_dict['image_1'][0].frequency_array>=self.minimum_frequency)&(self.ifos_dict['image_1'][0].frequency_array<=self.fid_fmax))[0]

        self.df = self.ifos_dict['image_1'][0].frequency_array[1] - self.ifos_dict['image_1'][0].frequency_array[0]

        self.quad_sum = quad_sum
        self.grid_choice = grid_choice
        self.fs = self.ifos_dict[f'image_1'].frequency_array[self.mask] ### assuming both images have same fs, minimum_frequency, maximum_frequency, sampling_frequency, duration

        self.fiducial_waveform = {}
        self.test_waveform = {}
        self.data = {}
        self.psd = {}

        for i in [1, 2]:


            self.fiducial_waveform[f'image_{i}'] = {}
            self.fiducial_waveform[f'image_{i}']['hL'], self.fiducial_waveform[f'image_{i}']['C'] = separate_l_likelihood.get_binned_detector_response(self.ifos_dict[f'image_{i}'], relby_jointpe_conversion(self.fiducial_model[f'image_{i}'], self.waveform_arguments), self.fs, self.waveform_arguments, self.mode_array)


            self.test_waveform[f'image_{i}'] = {}
            self.test_waveform[f'image_{i}']['hL'], self.test_waveform[f'image_{i}']['C']  = separate_l_likelihood.get_binned_detector_response(self.ifos_dict[f'image_{i}'], relby_jointpe_conversion(self.test_model[f'image_{i}'], self.waveform_arguments), self.fs, self.waveform_arguments, self.mode_array)

            self.data[f'image_{i}'] = np.array([ifo.frequency_domain_strain[self.mask] for ifo in ifos_dict[f'image_{i}']])
            self.psd[f'image_{i}'] = np.array([ifo.power_spectral_density_array[self.mask] for ifo in ifos_dict[f'image_{i}']])


        self.noise_log_l = self.noise_log_likelihood()
        self.binning_info = self.setup_bins()
    

        self.parameters = {}
        print('length of the adapted grid', len(self.binning_info[0]))
        print('length of the originl grid', len(self.fs))
        print('length of the grid proposed by bilby',len(self.ifos_dict['image_1'][0].frequency_array))

    def noise_log_likelihood(self):
        """
        Function computing the noise log likelihood for the two
        images at the same time
        """
        log_l = 0.

        # for the first image
        for i in [1, 2]:

            for ifo in self.ifos_dict[f'image_{i}']:
            # this is a duplication, check what's happening with this variable.
                mask = ifo.frequency_mask
                log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.ifos_dict[f'image_{i}'].duration)/2.

        return float(np.real(log_l))

    def log_likelihood_ratio(self):

        master_fbin_grid, master_fbin_ind, h_fbin, summary_data = self.binning_info


        p = relby_jointpe_conversion(self.parameters, self.waveform_arguments)

        plus_cross_dict = get_lensed_binned_detector_response(
            self.ifos_dict, p, master_fbin_grid, self.waveform_arguments, mode_array=self.mode_array)

        if plus_cross_dict['image_1']['hL'] is not None:
            r = {}
            s = {}

            log_likelihood_ratio = 0.

            for i in [1, 2]:
                r = (plus_cross_dict[f'image_{i}']['hL'])/(h_fbin[f'image_{i}']['hL'])
                s = (plus_cross_dict[f'image_{i}']['C'])/(h_fbin[f'image_{i}']['C'])

            
                r0, r1 = self.compute_bin_coefficients(master_fbin_grid, r) ## calculating 0th and first order coefficients
                C0, C1 = self.compute_bin_coefficients(master_fbin_grid, s) ##same as above

                Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1,
                                               summary_data[f'image_{i}']['A0'], summary_data[f'image_{i}']['A1'], summary_data[f'image_{i}']['B0'], summary_data[f'image_{i}']['B1'])
            
                log_likelihood_ratio += (Zdh-0.5*Zhh)

            return log_likelihood_ratio
        else:
            print('log likelihood is set to -np.inf since we couldnt gen wf')
            return -np.inf

    def log_likelihood(self):
        log_likelihood = self.log_likelihood_ratio() + self.noise_log_l
        return log_likelihood

    def compute_summary_data(self, fbin, Nbin, fbin_ind, fm, strain, signal, C0, psd, ndet):
        """
        fbin: relative binning grid
        Nbin: number of bins
        fbin_ind: relative binning grid index
        fm: mid values of the relative binning freqs
        strain: strain, image 1 or image 2
        signal: fiducial signal, image 1 or image 2
        psd: psd, image 1 or image 2
        """

        #ndet = len(self.ifos_image_1)

        Nmode = signal.shape[1]
        T = self.duration
        f = self.fs  # full frequency array

        A0 = 4./T*np.array([[[np.sum(strain[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(signal[i, l, fbin_ind[b]:fbin_ind[b+1]]*C0[i, l, fbin_ind[b]:fbin_ind[b+1]]) /
                           psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        A1 = 4./T*np.array([[[np.sum(strain[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(signal[i, l, fbin_ind[b]:fbin_ind[b+1]]*C0[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        B0 = 4./T*np.array([[[[np.sum(signal[i, l, fbin_ind[b]:fbin_ind[b+1]]*C0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(signal[i, ll, fbin_ind[b]:fbin_ind[b+1]]*C0[i, ll, fbin_ind[b]:fbin_ind[b+1]]) /
                           psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
        B1 = 4./T*np.array([[[[np.sum(signal[i, l, fbin_ind[b]:fbin_ind[b+1]]*C0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(signal[i, ll, fbin_ind[b]:fbin_ind[b+1]]*C0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]
                           * (f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])

        return {'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1}

    def setup_bins(self):

        master_fbin_grid = np.array([])
        master_fbin_ind = np.array([])


        for k in [1, 2]:
            fbin, fbin_ind, _ = self.get_binning_data(self.N_initial, image_index = k, quad_sum = self.quad_sum)
            print(f'grid length of image number {k}:', len(fbin))
            master_fbin_grid = np.union1d(fbin, master_fbin_grid)
            master_fbin_ind = np.union1d(fbin_ind, master_fbin_ind)

        master_fbin_ind = np.asarray(master_fbin_ind, dtype = int)
        
        print('total length after taking union of 4 images', len(master_fbin_grid))
        
        master_fm = (master_fbin_grid[1:] + master_fbin_grid[:-1])/2


        Nbin = len(master_fbin_ind)-1

        summary_data = {}
        h_fbin = {}
        print('Printing Grids')
        print(master_fbin_grid, '\n', master_fbin_ind, '\n', master_fm)

        for i in [1, 2]:
            ndet = len(self.ifos_dict[f'image_{i}'])
            summary_data[f'image_{i}'] = self.compute_summary_data(master_fbin_grid, Nbin, master_fbin_ind, master_fm, self.data[f'image_{i}'], self.fiducial_waveform[f'image_{i}']['hL'], self.fiducial_waveform[f'image_{i}']['C'], self.psd[f'image_{i}'], ndet)
            h_fbin[f'image_{i}'] = {}
            h_fbin[f'image_{i}']['hL'] = self.fiducial_waveform[f'image_{i}']['hL'][:, :, master_fbin_ind]
            h_fbin[f'image_{i}']['C'] = self.fiducial_waveform[f'image_{i}']['C'][:, :, master_fbin_ind]

        
        return master_fbin_grid, master_fbin_ind, h_fbin, summary_data

    def get_binning_data(self, N0, image_index, quad_sum, i = 0):
        """
        Computes the binning data needed to do mode by mode relative binning with scheme 2
        (fbin: array of frequencies of the bin edges (Hz)
         Nbin: number of bins
         fbin_ind: array of indices of bin edges in frequency array f
         fm: frequencies of the center of the bins (Hz))
        using the bisecting bin selection algorithm. This algorithm divides the region in 2 repeatedly, accepting bins when their error is below a threshold that achieves an overall target error for the test model. This algorithm iterates, changing the target number of bins until the threshold is achieved with the target number of bins.
        data: data object
        h0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
        h: test model modes, indices: detector, mode, frequency
        fmin: minimum frequency for bins (Hz)
        fmax: maximum frequency for the bins (Hz)
        N0: initial target number of bins
        E: total target error
        correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                       if False: set bound on error per bin assuming all bin errors are uncorrelated
        """
        print('bin progression', 'iteration', i , 'input bins', N0)

        if self.grid_choice=='bisection':
            print('Sorry that feature doesnt exist...exiting')
            exit()

        else: ## if grid choise is 'step' this code is executed
            fbin, Nbin, fbin_ind, fm = self.get_step_bin_grid(N0, image_index, quad_sum)

        print('output bins', Nbin, '###########\n')

        if Nbin==N0:
            return fbin, fbin_ind, fm

        else:
            return self.get_binning_data(Nbin, image_index, quad_sum, i = i+1)

    def compute_bin_error(self, f_lo, f_hi, image_index, Cfbin=None):
        """
        Computes the error contribution to the log likelihood of mode-by-mode relative binning scheme 2 for one bin that begins at the frequency in the data object closest to f_lo and ends at the frequency in the data object closest to f_hi.
        data: data object
        h0: fiducial model modes, indices: detector, mode, frequency
        h: test model modes, indices: detector, mode, frequency
        f_lo: target lower frequency for the bin, this exact frequency may not be in the frequency array
        f_hi: target upper frequency for the bin, this exact frequency may not be in the frequency array
        sign: boolean, if True: returns the error (relative binning - exact) with its sign, this option is used for investigating error in relative binning
                       if False: returns the absolute value of the error, this option is used in the bisecting bin selection algorithm
        
        Cfbin: set to None when simultaneously in hL and C. Mention the Value when rebinning in C
        """
        # build length 2 array for small bin
        
        ii = image_index
        ndet = len(self.ifos_dict[f'image_{ii}'])
        fbin = np.array([f_lo, f_hi])
        fbin_ind = np.unique(np.argmin(np.abs(self.fs[:, np.newaxis] - fbin), axis=0))
        fbin = self.fs[fbin_ind]
        Nbin = 1
        fm = 0.5*(fbin[1:]+fbin[:-1])
        
        summary_data = self.compute_summary_data(fbin, Nbin, fbin_ind, fm, self.data[f'image_{ii}'], self.fiducial_waveform[f'image_{ii}']['hL'], self.fiducial_waveform[f'image_{ii}']['C'], self.psd[f'image_{ii}'], ndet)
        r = (self.test_waveform[f'image_{ii}']['hL']/self.fiducial_waveform[f'image_{ii}']['hL'])[:,:,fbin_ind]

        s = (self.test_waveform[f'image_{ii}']['C']/self.fiducial_waveform[f'image_{ii}']['C'])[:,:,fbin_ind]
            
        t_allmodes = np.sum(self.test_waveform[f'image_{ii}']['C']*self.test_waveform[f'image_{ii}']['hL'], axis = 1)
            
        exact_lnL = np.sum([inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.data[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.psd[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.duration) - 1/2*inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.psd[f'image_{ii}'][i, fbin_ind[0]:fbin_ind[1]], self.duration) for i in range(ndet)])
        
        r0, r1 = self.compute_bin_coefficients(fbin, r)
        C0, C1 = self.compute_bin_coefficients(fbin, s)
       
        Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, summary_data['A0'], summary_data['A1'], summary_data['B0'], summary_data['B1'])
        
        rb_lnL = Zdh - 1/2*Zhh
       

        return np.abs(rb_lnL - exact_lnL)

    

    def compute_overlaps(self, r0, r1, C0, C1, A0, A1, B0, B1):
        """
        Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf) used in advanced mode by mode relative binning (Scheme 1).
        r0, r1: linear coefficients for the ratio of time-dependent L-frame modes over fiducial time-dependent L-frame modes, indices: detector, mode, frequency
        C0, C1: linear coefficients for the coefficient of the time-dependent L-frame modes, indices: detector, mode, frequency
        A0, A1, B0, B1: summary data for mode-by mode relative binning, A indices: detector, mode, frequency, B indices: detector, mode, mode, frequency
        """
        Zdh = np.real(np.sum( A0*np.conj(r0)*np.conj(C0) + A1*( (np.conj(r0)*np.conj(C1)) + (np.conj(r1)*np.conj(C0)) ) ))
        
        Zhh = np.real(np.sum( B0*r0[:, np.newaxis, :, :]*C0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*np.conj(C0[:, :, np.newaxis, :]) 
                            + B1*( r0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*(np.conj(C0[:, :, np.newaxis, :])*C1[:, np.newaxis, :, :] + C0[:, np.newaxis, :, :]*np.conj(C1[:, :, np.newaxis, :])) + C0[:, np.newaxis, :, :]*np.conj(C0[:, :, np.newaxis, :])*(np.conj(r0[:, :, np.newaxis, :])*r1[:, np.newaxis, :, :] + r0[:, np.newaxis, :, :]*np.conj(r1[:, :, np.newaxis, :])) ) ))
        
        return Zdh, Zhh

    def compute_bin_coefficients(self, fbin, r):
        """
        Computes the bin coefficients used in mode-by-mode relative binning.
        fbin: array of frequencies of the bin edges (Hz)
        r: array of ratios of model over fiducial model, indices: detector, mode, frequency
        """
        binwidths = fbin[1:] - fbin[:-1]
        
        # rplus: right edges of bins
        rplus = r[:, :, 1:]
        # rminus: left edges of bins
        rminus = r[:, :, :-1]
        
        r0 = 0.5 * (rplus + rminus)
        r1 = (rplus - rminus) / binwidths[np.newaxis, np.newaxis, :]
        
        return r0, r1



    def get_step_bin_grid(self, N, image_index, quad_sum):
        
        if quad_sum:
            print('quad sum')
            Emax = self.total_err/np.sqrt(N)
        else:
            print('direct sum')
            Emax = self.total_err/N
        print('Emax of the loop', Emax)
        
        output_step_bins = [self.fs[0]]
        output_step_bins_index = [0]
        i = 0
        while i <len(self.fs):
            #print('step flow bin index', i)
            j = 1 # steps in which the bins will increase
            while (i+j)<len(self.fs):
                #print('step adding bin', j)
                flo = self.fs[i]
                fhi = self.fs[i+j]
                add_bins = i+j
                #print('flo and fhi', flo, fhi)
                bin_error = self.compute_bin_error(flo, fhi, image_index)
                j+=1

                if bin_error>Emax:
                    i = i+j-1
                    #print('we reached', i+j-1)
                    output_step_bins.append(self.fs[add_bins])
                    output_step_bins_index.append(add_bins)

                    #print('number of bins combined', j)
                    #print('moving to the next bin \n\n')
                    break
            if (i+j)==len(self.fs):
                print('Breaking because we reached the end')
                break

        output_step_bins = np.unique(np.array(output_step_bins))
        Nbin = len(output_step_bins)
        output_step_bins_index = np.unique(np.array(output_step_bins_index))
        fm = (output_step_bins[1:] + output_step_bins[:-1])/2
        return np.unique(np.array(output_step_bins)), len(output_step_bins)-1, np.unique(np.array(output_step_bins_index)), fm



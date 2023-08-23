import bilby 
from tqdm import tqdm
from bilby.core.likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
from bilby.gw.utils import noise_weighted_inner_product
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt
from time import time
from . import xphm 
from numba import njit


def no_jit_interp(fs, rebinned_c, Ctotal, num_modes, num_ifos, Creturn):
    
    Creturn = np.array([[np.interp(fs, rebinned_c, Ctotal[i, j, :]) for j in range(num_modes)] for i in range(num_ifos)])
    return Creturn

@njit
def jit_interp(fs, rebinned_c, Ctotal, num_modes, num_ifos, Creturn):
    '''
    this function works faster without jit, or as fast probably
    '''
    for i in range(num_ifos):
        for j in range(num_modes):
            Creturn[i, j, :] = np.interp(fs, rebinned_c, Ctotal[i, j, :])
    return Creturn


@njit
def compute_numba_overlaps(r0, r1, C0, C1, A0, A1, B0, B1):
    

    m = C0.shape[2]
    Zdh = A0*np.conj(r0)*np.conj(C0) + A1*(np.conj(r0)*np.conj(C1) + np.conj(r1)*np.conj(C0))



    C0a = C0.reshape(3, 1, 5, m)
    C0b = C0.reshape(3, 5, 1, m)

    r0a = r0.reshape(3, 1, 5, m)
    r0b = r0.reshape(3, 5, 1, m)

    term_1 = B0*r0a*C0a*np.conj(r0b)*np.conj(C0b)
    term_2 = r0a*np.conj(r0b)*(np.conj(C0b)*C1.reshape(3, 1, 5, m)
                             + C0a*np.conj(C1.reshape(3, 5, 1, m))) 


    term_3 = C0a*np.conj(C0b)*(np.conj(r0b)*r1.reshape(3, 1, 5, m) + r0a*np.conj(r1.reshape(3, 5, 1, m)))
    

    Zhh = term_1+(B1*(term_2+term_3))
            
    return np.real(np.sum(Zdh)), np.real(np.sum(Zhh))

def relby_conversion(parameters, waveform_arguments):



    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'], 
                                                                                        parameters['mass_ratio'])

    iota, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(parameters['theta_jn'], parameters['phi_jl'], parameters['tilt_1'], parameters['tilt_2'],
                                                                                                                        parameters['phi_12'], parameters['a_1'], parameters['a_2'], 
                                                                                                                        parameters['mass_1']* lal.MSUN_SI, parameters['mass_2']* lal.MSUN_SI,
                                                                                                                                                    waveform_arguments['reference_frequency'], parameters['phase'])

    relby_parameters = {'mass_1': parameters['mass_1'], 'mass_2': parameters['mass_2'],
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
        'geocent_time': parameters['geocent_time']}

    
    relby_parameters['theta_jn'] = xphm.theta_jn_func_op(relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'], 
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'], 
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'], waveform_arguments['reference_frequency'])



    relby_parameters['kappa']= xphm.kappa_func_op(relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'], 
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'], 
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'], waveform_arguments['reference_frequency'])

    

    return relby_parameters

def lal_f_max(parameters, reference_frequency=50.):

   
    
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

def get_binned_detector_response(ifos, parameters, fs, waveform_arguments, mode_array, rebinned_c = None):

    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'], parameters['mass_ratio'])
    

    


    #t=time()
    #hL = xphm.compute_td_L_frame_modes(parameters, fs, waveform_arguments, np.array(mode_array))
    #print('time taken to get L frame', time()-t)

    #Cfreq = np.random.choice(fs.data, int(len(fs)/10))
    #print('computing C on {} points'.format(len(Cfreq)))

    #t = time()
    if rebinned_c is not None:
        Cplus, Ccross, hL = xphm.compute_C_prefactors(parameters, rebinned_c, waveform_arguments, np.array(mode_array), fs)
        #Cplus = np.interp(fs.data, rebinned_c, Cplus)
        #Ccross = np.interp(fs.data, rebinned_c, Ccross)


    else:
        Cplus, Ccross, hL = xphm.compute_C_prefactors(parameters, fs, waveform_arguments, np.array(mode_array), fs)

    #print('time taken to get C', time()-t)
    
    #t = time()
    #Cplus = [np.interp(fs.data, Cfreq, _Cplus[i, :]) for i in range(len(mode_array))]
    #Ccross = [np.interp(fs.data, Cfreq, _Ccross[i, :]) for i in range(len(mode_array))]
    #print('time taken to interpolate C', time()-t)

    Fplus = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'], 
        parameters['geocent_time'], parameters['psi'], 'plus') for ifo in ifos])
    Fcross = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'], 
        parameters['geocent_time'], parameters['psi'], 'cross') for ifo in ifos])


    dt_geocent = np.array([ifo.time_delay_from_geocenter(parameters['ra'], parameters['dec'], parameters['geocent_time']) for ifo in ifos])
    dt_total = dt_geocent+(parameters['geocent_time']-ifos[0].start_time)   ## this is basically adding time delay from geocent and duration of the signal

    #print('dt_total', dt_total)
    #print('shape before', np.shape(Fplus))
    Fplus = Fplus[:, np.newaxis, np.newaxis]
    #Fplus = Fplus[:, np.newaxis]
    Fcross = Fcross[:, np.newaxis, np.newaxis]
    #print('shpae after', np.shape(Fplus))


    Cplus_projected = Fplus*Cplus #np.array([[Fplus[j]*Cplus[i] for i in range(len(mode_array))] for j in range(len(ifos))])
    Ccross_projected = Fcross*Ccross #np.array([[Fcross[j]*Ccross[i] for i in range(len(mode_array))] for j in range(len(ifos))])



    Ctotal = Cplus_projected+Ccross_projected
    num_ifos = len(ifos)
    num_modes = len(mode_array)
    if rebinned_c is not None:
        Creturn = np.zeros([len(ifos), len(mode_array), len(fs)], dtype = complex)
        
        Ctotal = jit_interp(fs, rebinned_c, Ctotal, num_modes, num_ifos, Creturn) #np.array([[np.interp(fs, rebinned_c, Ctotal[j, i, :]) for i in range(len(mode_array))] for j in range(len(ifos))])
    

    exp_timeshift = np.exp(-1j*2*np.pi*fs*dt_total[:, np.newaxis])
    if hL is not None:
        hL = exp_timeshift[:, np.newaxis, :]*hL[np.newaxis, :, :]

    #print('shapes', np.shape(hL), np.shape(Ctotal))
    #hL = np.array([[np.exp(-1j*2*np.pi*fs*dt_total[j])*hL[i] for i in range(len(mode_array))] for j in range(len(ifos))])
    #hL = np.exp(-1j*2*np.pi*fs[:, np.newaxis]*dt_total)[:, :, np.newaxis]*hL[np.newaxis, :, :]
    return hL, Ctotal


def get_hplus_hcross(parameters, fs, waveform_arguments, mode_array):


    #t=time()
    hL = xphm.compute_td_L_frame_modes(parameters, fs, waveform_arguments, np.array(mode_array))
    #print('time taken to get L frame', time()-t)

    #Cfreq = np.random.choice(fs.data, int(len(fs)/10))
    #print('computing C on {} points'.format(len(Cfreq)))

    #t = time()
    Cplus, Ccross = xphm.compute_C_prefactors(parameters, fs, waveform_arguments, np.array(mode_array))


    hplus = np.sum(hL*Cplus, axis = 0)
    hcross = np.sum(hL*Ccross, axis = 0)

    return {'plus': hplus, 'cross': hcross}    

def inn_prod(x, y, psd, T):
    """
    Computes the inner product between two frequency series.
    x, y: frequency series arrays
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    """
    return np.real(np.sum(4*np.conj(x)*y/psd/T))

class RelativeBinningHOM(Likelihood):
    """
    Class to construct likelihood with relative binning with higher order modes
    """
    def __init__(self, interferometers, fiducial_model, test_model, priors, fid_fmax, mode_array, waveform_arguments, total_err, rebin_c, grid_choice, simulation=True):
        
        """
        Initialization of the class.

        ARGS:
        -----
        - interferometers: list of interferometers that contain the signal
        - waveform_generator: the bilby waveform generator object that should
                              be used to setup the run
        - fiducial_model: dictionary with the fiducial model parameters to be used
                 for the analysis
        - test_model: dictionary of the test model parameters to be used to make bins. This model will only be used to make 
                        bins not for calculating likelihood later
        - priors: the priors used for the different parameters
        - total_err: total error on the rb likelihood compared to true likelihood
        - rebin c: True if you want to rebin c
        """
        
        super(RelativeBinningHOM, self).__init__(dict())
        self.interferometers = interferometers
        self.priors = priors
        self.fid_fmax = np.min([fid_fmax, self.interferometers[0].frequency_array[-1]])
        #print('fmax used', self.fid_fmax, self.interferometers[0].frequency_array[-1], 'comparison')
        print('fmax of the analysis', self.fid_fmax)
        self.mask = np.where((self.interferometers[0].frequency_array>=waveform_arguments['minimum_frequency'])&(self.interferometers[0].frequency_array<=self.fid_fmax))[0]
        #print('fmax', self.fid_fmax)

        self.fs = self.interferometers[0].frequency_array[self.mask]
        self.waveform_arguments = waveform_arguments

        self.df = self.interferometers[0].frequency_array[1]-self.interferometers[1].frequency_array[0]
        self.total_err = total_err
        self.duration = self.interferometers[0].duration #duration
        self.sampling_frequency = self.waveform_arguments['sampling_frequency']
        self.minimum_frequency = self.waveform_arguments['minimum_frequency']
        self.N_initial = 0.01*len(self.fs) ## what is the purpose of this?
        self.fiducial_model = relby_conversion(fiducial_model, self.waveform_arguments)
        self.mode_array = mode_array
        self.quad_sum = True
        self.h0, self.hC0 = get_binned_detector_response(self.interferometers, self.fiducial_model, self.fs, self.waveform_arguments, self.mode_array)

        self.test_model = relby_conversion(test_model, self.waveform_arguments)
        self.t0, self.tC0 = get_binned_detector_response(self.interferometers, self.test_model, self.fs, self.waveform_arguments, self.mode_array)
        self.data = np.array([ifo.frequency_domain_strain[self.mask] for ifo in self.interferometers])
        self.psd_array = np.array([ifo.power_spectral_density_array.copy()[self.mask] for ifo in self.interferometers])
        '''if 'minimum_frequency' in self.waveform_generator.waveform_arguments.keys():
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
                        '''
        self.grid_choice = grid_choice
        self.noise_log_l = self.noise_log_likelihood()
        self.binning_info = self.setup_bins()
        #exit()
        print('length of rb bins', len(self.binning_info[0]))

        self.threshold = self.total_err/len(self.binning_info[0]) ### notice that this is still hardcoded
        print('threshold value used to rebin C', self.threshold)
        self.rebin_c = rebin_c

        self.bug_fix = self.fiducial_log_likelihood()
        if self.rebin_c:

            self.sparse_C_bins = self.rebin_c_prefactors(self.binning_info[0], self.binning_info[1], self.threshold)

            print('sparse c bins', self.sparse_C_bins)
            print('length of c bins', len(self.sparse_C_bins))
        else:
            self.sparse_C_bins = None

        self.parameters = {}
        print('length of the adapted grid', len(self.binning_info[0]))
        print('length of the originl grid', len(self.fs))
        print('length of the grid proposed by bilby', len(self.interferometers[0].frequency_array))


    def noise_log_likelihood(self):
        """
        Fuction computing the noise log likelihood. 
        This is the same as for the usual runs
        """
        log_l = 0.

        for ifo in self.interferometers:
            ### note that noise log likelihood is evaluated on full freq grid 
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask], 
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.duration)/2.

        return float(np.real(log_l))

    def fiducial_log_likelihood(self):
        fbin, f_bin_ind, h0_fbin, hC0_fbin, A0, A1, B0, B1 = self.binning_info


        p = self.fiducial_model#relby_jointpe_conversion(self.fiducial_model, self.waveform_arguments)

        h, hC = get_binned_detector_response(
            self.interferometers, p, fbin, self.waveform_arguments, mode_array=self.mode_array, rebinned_c=None)

        if h is not None:
            # print(time()-t, 'time taken to generate hl and C\n\n\n')
            # print(np.shape(fbin), np.shape(hC), 'shapeo fo fbin and hC')

            r = (h/h0_fbin)

            s = hC/hC0_fbin

            image_1_r0, image_1_r1 = self.compute_bin_coefficients(fbin, r)
            image_1_C0, image_1_C1 = self.compute_bin_coefficients(fbin, s)

            Zdh1, Zhh1 = self.compute_overlaps(image_1_r0, image_1_r1, image_1_C0, image_1_C1,
                                               A0, A1, B0, B1)
            

            log_likelihood_ratio_1 = Zdh1-0.5*Zhh1
            print('Zdh and Zhh at fiducial value', Zdh1, Zhh1)
            #print(log_likelihood_ratio_1, log_likelihood_ratio_2, 'individual log l ratio inside')
            return Zdh1- (0.5*(Zhh1)) + self.noise_log_l
        else:
            print('log likelihood is set to -np.inf since we couldnt gen wf')
            return -np.inf

    def log_likelihood_ratio(self):
       
        fbin, f_bin_ind, h0_fbin, hC0_fbin, A0, A1, B0, B1 = self.binning_info
        
        #t = time()
        #relby_readable_proposal = relby_conversion(self.parameters, 20, 50, self.sampling_frequency, self.duration)
        #print(time()-t, 'time taken to convert to relby params')
        
        #t = time()



        p = relby_conversion(self.parameters, self.waveform_arguments)

        h, hC = get_binned_detector_response(self.interferometers, p, fbin, self.waveform_arguments, mode_array = self.mode_array, rebinned_c = self.sparse_C_bins)
        

        if h is not None:
            #print(time()-t, 'time taken to generate hl and C\n\n\n')
            #print(np.shape(fbin), np.shape(hC), 'shapeo fo fbin and hC')
            r = (h/h0_fbin)
            s = (hC/hC0_fbin)
            r0, r1 = self.compute_bin_coefficients(fbin, r)
            C0, C1 = self.compute_bin_coefficients(fbin, s)

            #t = time()
            #Zdh, Zhh = compute_numba_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
            #print(time()-t, 'time for numba overlaps')
            #t = time()
            Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
            #print(time()-t, 'time for self overlaps')
            return Zdh - (Zhh/2.)
        else:
            return -np.inf


    def log_likelihood(self):
        log_likelihood = self.log_likelihood_ratio() + self.noise_log_l
        return log_likelihood



    def compute_summary_data(self, fbin, Nbin, fbin_ind, fm):
        """
        Computes the summary data used in mode-by-mode relative binning (Scheme 1).
        d: data frequency series array
        hhat0: fiducial model time-dependent L-frame mode frequency series array
        psd: psd array
        T: length of time series (s)
        f: full frequency array (Hz)
        fbin: array of frequencies of the bin edges (Hz)
        Nbin: number of bins
        fbin_ind: array of indices of bin edges in frequency array f
        fm: frequencies of the center of the bins (Hz)
        """
        

        ndet = len(self.interferometers)
        Nmode = self.h0.shape[1]
        T = self.duration
        f = self.fs ## full frequency array 

        A0 = 4./T*np.array([[[np.sum(self.data[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]])*np.conj(self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        A1 = 4./T*np.array([[[np.sum(self.data[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]])*np.conj(self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
        B0 = 4./T*np.array([[[[np.sum(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])*np.conj(self.hC0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
        B1 = 4./T*np.array([[[[np.sum(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])*np.conj(self.hC0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
    
        '''A0 = 4/T*np.array([[[np.sum(self.data[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(nmode)] for i in range(ndet)])
                                A1 = 4/T*np.array([[[np.sum(self.data[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(nmode)] for i in range(ndet)])
                                B0 = 4/T*np.array([[[[np.sum(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(nmode)] for ll in range(nmode)] for i in range(ndet)])
                                B1 = 4/T*np.array([[[[np.sum(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(nmode)] for ll in range(nmode)] for i in range(ndet)])
                        '''
        return A0, A1, B0, B1


    def setup_bins(self):
        """
        Compute the binning data for relative binning. This only needs to be done once for a particular fiducial model. The results are used in computing the likelihood with relative binning for a given model.
        data: data object
        fiducial_model: fiducial model object, must have compute_modes = True
        modebymode: boolean, if True: computes the binning data for relative binning mode by mode
                               if False: computes the binning data for standard modeless relative binning
        test_model: test model object required for bisecting bin selection algorithm
        Etot: total error target for bisecting bin selection algorithm
        correlated_bin_error: boolean, used in bisecting bin selection algorithm, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                                            if False: set bound on error per bin assuming all bin errors are uncorrelated
        scheme: an int that specifies the scheme to be used, acceptable values are 1 and 2, 1 is used by default
        """
        #T = data.T
        #f = data.fs
        
        #fmin = fiducial_model.fmin
        #fmax = fiducial_model.fmax
        
        

        fbin, Nbin, fbin_ind, fm = self.get_binning_data_bisect(self.N_initial, self.quad_sum)
        '''
        plt.scatter(fbin, np.zeros(len(fbin)), label = 'adapted grid', s = 1)
        plt.scatter(self.fs, np.ones(len(self.fs)), label = 'original grid', s = 1)
        plt.ylim(-2, 2)
        plt.legend()
        plt.show()
        
        fbin = self.fs
        Nbin = len(self.fs)
        fbin_ind = np.arange(Nbin)
        fm = (fbin[1:] + fbin[:-1])/2
        '''
        self.h0_fbin = self.h0[:, :, fbin_ind]
        self.hC0_fbin = self.hC0[:, :, fbin_ind]

        A0, A1, B0, B1 = self.compute_summary_data(fbin, Nbin, fbin_ind, fm)
        return fbin, fbin_ind, self.h0_fbin, self.hC0_fbin, A0, A1, B0, B1, 


    def get_binning_data_bisect(self, N0, quad_sum, i = 0):
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
        if self.grid_choice=='bisection':
            fbin, Nbin, fbin_ind, fm = self.get_binning_data_bisect_iteration(N0)
        else:
            fbin, Nbin, fbin_ind, fm = self.get_step_bin_grid(N0, quad_sum = quad_sum)

        #fbins = self.get_step_bin_grid(N0)
        #Nbin = len(fbins)
        print('bin propagation and i ', Nbin, N0, i)
        if i==40:
            print('breakign because iterations reached', i)
            return fbin, Nbin, fbin_ind, fm


        if Nbin==N0:
            if Nbin<40:
                 print('Too few bids setting quad sum to False')
                 return self.get_binning_data_bisect(Nbin, quad_sum = False, i = i+1)
            else:
                return fbin, Nbin, fbin_ind, fm

        else:
            return self.get_binning_data_bisect(Nbin, quad_sum = quad_sum, i = i+1)


    def get_binning_data_bisect_iteration(self, N):
        """
        Helper function for get_binning_data_bisect_2. This function does bisection checks trying to achieve the target error E in N bins. It may require more bins or fewer bins. If so, this function is called again.
        data: data object
        h0: fiducial model modes, indices: detector, mode, frequency
        h: test model modes, indices: detector, mode, frequency
        fmin: minimum frequency for bins (Hz)
        fmax: maximum frequency for the bins (Hz)
        N: target number of bins
        E: total target error
        correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                       if False: set bound on error per bin assuming all bin errors are uncorrelated
        """
        print('Using bisection grid to make bins')
        fmin, fmin_ind = self.nearest_freq(self.fs, self.fs[0])
        fmax, fmax_ind = self.nearest_freq(self.fs, self.fs[-1])
        max_bisections = int(np.floor(np.log2(fmax_ind-fmin_ind)))
        Emax = self.total_err/np.sqrt(N)
        print('Emax of the iteration, divide by sqrt N', Emax)
        fbin_nomin, Nbin, fbin_ind_nomin = self.bisect_bin_search(f_lo = fmin, f_hi = fmax, depth = 0, max_depth = max_bisections, Emax = Emax) 
        
        fbin = np.append(fmin, fbin_nomin)
        fbin_ind = np.append(fmin_ind, fbin_ind_nomin)
        fm = (fbin[1:] + fbin[:-1])/2
        return fbin, Nbin, fbin_ind, fm

    def bisect_bin_search(self, f_lo, f_hi, depth, max_depth, Emax):
        """
        Helper function for get_binning_data_bisect_2_iteration. This recursive function does the bisection until either the target is achieved or the maximum bisection depth is reached.
        data: data object
        h0: fiducial model modes, indices: detector, mode, frequency
        h: test model modes, indices: detector, mode, frequency
        f_lo: lower frequency for a test bin (Hz)
        f_hi: upper frequency for a test bin (Hz)
        depth: number of times bisection has been performed
        maxdepth: maximum number of times bisection is to be performed before the bisection stops
        Emax: maximum error allowed per bin (unless maxdepth is reached)
        """
        if (f_hi-f_lo)<=(self.df*2):
            fmax, fmax_ind = self.nearest_freq(self.fs, f_hi)
            return np.array([fmax]), 1, np.array([fmax_ind])
        bin_error = self.compute_bin_error(f_lo, f_hi)
        #print('likelihood error', bin_error)
        #bin_match_error = self.compute_bin_error_with_match(f_lo, f_hi)
        #print('bin_match_error', bin_match_error, '\n')
        if bin_error < Emax:
            #print('hit error', bin_error, Emax)
            #print('flo and fhi', f_lo, f_hi)
            fmax, fmax_ind = self.nearest_freq(self.fs, f_hi)
            return np.array([fmax]), 1, np.array([fmax_ind])
        else:
            f_mid = (f_lo + f_hi)/2
            fbin_lo, Nbin_lo, fbin_ind_lo = self.bisect_bin_search(f_lo, f_mid, depth+1, max_depth, Emax)
            fbin_hi, Nbin_hi, fbin_ind_hi = self.bisect_bin_search(f_mid, f_hi, depth+1, max_depth, Emax)
            fbin = np.append(fbin_lo, fbin_hi)
            Nbin = Nbin_lo + Nbin_hi
            fbin_ind = np.append(fbin_ind_lo, fbin_ind_hi)
            return fbin, Nbin, fbin_ind

    def compute_bin_error(self, f_lo, f_hi, Cfbin=None):
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
        
        ndet = len(self.interferometers)
        fbin = np.array([f_lo, f_hi])
        fbin_ind = np.unique(np.argmin(np.abs(self.fs[:, np.newaxis] - fbin), axis=0))
        fbin = self.fs[fbin_ind]
        Nbin = 1
        fm = 0.5*(fbin[1:]+fbin[:-1])
        
        # compute relative binning log likelihood contribution for the given bin
        A0, A1, B0, B1 = self.compute_summary_data(fbin, Nbin, fbin_ind, fm)
        
        r = (self.t0/self.h0)[:,:,fbin_ind]
        s = (self.tC0/self.hC0)[:, :, fbin_ind]
        if Cfbin is None:
            Cfbin = self.tC0[:,:,fbin_ind]
            #print('when not interpolating', Cfbin)
        else:
            Cfbin = Cfbin
        
        r0, r1 = self.compute_bin_coefficients(fbin, r)
        C0, C1 = self.compute_bin_coefficients(fbin, s)

        #Zdh, Zhh = compute_numba_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
        #print(Zhh)

        Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
        #print(Zhh, '\n\n')
        rb_lnL = Zdh - 1/2*Zhh
        
        t_allmodes = np.sum(self.tC0*self.t0, axis = 1)
        
        exact_lnL = np.sum([inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.data[i, fbin_ind[0]:fbin_ind[1]], self.psd_array[i, fbin_ind[0]:fbin_ind[1]], self.duration) - 1/2*inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]], t_allmodes[i, fbin_ind[0]:fbin_ind[1]], self.psd_array[i, fbin_ind[0]:fbin_ind[1]], self.duration) for i in range(ndet)])
        #print('r0, r1, c0, c1', r0, r1, C0, C1)
       
        return np.abs(rb_lnL - exact_lnL)

    def nearest_freq(self, f, target_f):
        """
        Returns the frequency in f closest to target_f.
        f: frequency array
        target_f: target frequency
        """
        f_ind = np.argmin(np.abs(f-target_f))
        f_near = f[f_ind]
        return f_near, f_ind

    

    def compute_overlaps(self, r0, r1, C0, C1, A0, A1, B0, B1):
        """
        Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf) used in advanced mode by mode relative binning (Scheme 1).
        r0, r1: linear coefficients for the ratio of time-dependent L-frame modes over fiducial time-dependent L-frame modes, indices: detector, mode, frequency
        C0, C1: linear coefficients for the coefficient of the time-dependent L-frame modes, indices: detector, mode, frequency
        A0, A1, B0, B1: summary data for mode-by mode relative binning, A indices: detector, mode, frequency, B indices: detector, mode, mode, frequency
        """
        
        Zdh = np.real(np.sum( A0*np.conj(r0)*np.conj(C0) 
                            + A1*( (np.conj(r0)*np.conj(C1)) + (np.conj(r1)*np.conj(C0)) ) ))
        
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



    def rebin_c_prefactors(self, freq_joint_bins, idx_joint_bins, threshold):
        output_c_bins = [freq_joint_bins[0]]
        idx_ouput_c_bins = []
        i = 0


        while i < len(idx_joint_bins):
            print('f_low bin index i ################', i)
            j = 1 ## steps in which the bins will increase
            while (i+j)<len(idx_joint_bins): ### while the low bin index (i) + number of steps are less than total number of bins
                print('printing j', j)

                add_bins = i+j ## number of bins added to the ith bin  ## this is the index of my top bin
                
                Cfbin = self.tC0[:, :, [idx_joint_bins[i], idx_joint_bins[add_bins]]]  ## find the C values on the edges, 
                f_bins_inbetween = freq_joint_bins[i:(add_bins+1)] ## freqs of the bin inbetween and left and right edges. C will be interpolated on these points
                C_interpolated = np.array([[np.interp(f_bins_inbetween, [f_bins_inbetween[0], f_bins_inbetween[-1]], Cfbin[i, j, :]) for j in range(len(self.mode_array))] for i in range(len(self.interferometers))])

                #print('shape of the interpolated C', np.shape(C_interpolated))
                #print('f_bins inbetween', f_bins_inbetween)
                #print('sliced', C_interpolated[:, :, 0])
                #print('transposed', np.transpose([C_interpolated[:, :, 0], C_interpolated[:, :, 0+1]]))
                #print('the value that goes into function', C_interpolated)
                
                C_interpolate_error = [self.compute_bin_error(f_bins_inbetween[k], f_bins_inbetween[k+1], C_interpolated[:, :, [k, k+1]]) for k in range(len(f_bins_inbetween)-1)]
                C_exact_error =  [self.compute_bin_error(f_bins_inbetween[k], f_bins_inbetween[k+1]) for k in range(len(f_bins_inbetween)-1)]
                
                #print('bin error when C is interpolated')
                
                C_interpolate_error_total = np.sum(C_interpolate_error)
                print(C_interpolate_error_total)
                C_exact_error_total = np.sum(C_exact_error)

                #print(C_interpolate_error_total, C_exact_error_total)
                #print('interpolate and exact erro')
                j+=1
                if C_interpolate_error_total<=threshold:#C_exact_error_total: ########## WARNING: The number 1e-3 is hardcoded at the moment. This will have a major impact on speed vs accuracy.
                    #j+=1

                    print('adding a bin')
                else:
                    i = i+j-1
                    output_c_bins.append(freq_joint_bins[add_bins])
                    print('number of bins combined $$$$$$$$$$$$$$$', j)

                    print('move to next bin\n\n\n')
                    break
                #print('breaking the loop because i+j sum is complete', i+j)

            if (i+j)==len(idx_joint_bins):
                print('breaking because the last bin was merged and we reached the edge')
                print('Ending the rebinning of C')
                output_c_bins.append(freq_joint_bins[-1])
                break

        return np.unique(np.array(output_c_bins))


    def get_step_bin_grid(self, N, quad_sum):
        print('using step grid to make bins, N div')
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
        while i < len(self.fs):
            #print('step flow bin index', i)
            j = 1 # steps in which the bins will increase
            while (i+j)<len(self.fs):
                #print('step adding bin', j)
                flo = self.fs[i]
                fhi = self.fs[i+j]
                add_bins = i+j
                #print('flo and fhi', flo, fhi)
                bin_error = self.compute_bin_error(flo, fhi)
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




    def compute_bin_error_with_match(self, f_lo, f_hi):
        """
        This function will estimate the error by calculating match instead of the likelihood
        Exactly like compute_bin_error. But uses match instead of likelihood. 
        This function is not useful. 
        Calculating error on likelihood is more robust than calculating error on waveform match.
        match = 4 * delta_f * exact_h * rb_interpolated_h
        """

        ndet = len(self.interferometers)

        fbin = np.array([f_lo, f_hi])
        fbin_ind = np.unique(np.argmin(np.abs(self.fs[:, np.newaxis] - fbin), axis=0))
        
        f_band = self.fs[fbin_ind[0]:fbin_ind[1]]

        fbin = self.fs[fbin_ind]
        Nbin = 1
        fm = 0.5*(fbin[1:]+fbin[:-1])

        exact_h0 = self.h0[:, :, fbin_ind[0]:fbin_ind[1]]
        exact_hC0 = self.hC0[:, :, fbin_ind[0]:fbin_ind[1]]

        rb_h0 = self.h0[:, :, fbin_ind]
        rb_hC0 = self.hC0[:, :, fbin_ind]


        interp_rb_h0 = np.array([np.array([(np.interp(f_band, fbin, rb_h0[i, j, :])) for j in range(len(self.mode_array))]) for i in range(ndet)])
        interp_rb_hC0 = np.array([np.array([np.interp(f_band, fbin, rb_hC0[i, j, :]) for j in range(len(self.mode_array))]) for i in range(ndet)])

        #print('rb_h0, rb_hC0', interp_rb_h0, interp_rb_hC0)

        full_rb_h = np.sum(interp_rb_h0*interp_rb_hC0, axis = 1) ## full rb waveform for all dets
        full_exact_h = np.sum(exact_h0*exact_hC0, axis = 1) ### full exact waveform for all detectors




        max_match = np.sum(full_exact_h[0]*np.conj(full_exact_h[0]))

        norm = np.sum(full_rb_h[0]*np.conj(full_rb_h[0]))
        approx_match = np.sum(full_exact_h[0]*np.conj(full_rb_h[0]))/np.sqrt(max_match)/np.sqrt(norm)


        #print('approx match', approx_match)
        return approx_match






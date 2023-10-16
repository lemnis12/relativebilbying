import bilby 
from tqdm import tqdm
from bilby.core.likelihood import Likelihood
import numpy as np
from bilby.gw.utils import noise_weighted_inner_product
import lal
from . import xphm


def relby_conversion(parameters, waveform_arguments):



    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        parameters['chirp_mass'], parameters['mass_ratio'])

    iota, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        parameters['theta_jn'], parameters['phi_jl'], parameters['tilt_1'], parameters['tilt_2'],
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

    
    relby_parameters['theta_jn'] = xphm.theta_jn_func_op(
        relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'], 
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'], 
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'], waveform_arguments['reference_frequency'])

    relby_parameters['kappa']= xphm.kappa_func_op(
        relby_parameters['mass_1'], relby_parameters['mass_2'], relby_parameters['chi_1x'], 
        relby_parameters['chi_1y'], relby_parameters['chi_1z'], relby_parameters['chi_2x'], relby_parameters['chi_2y'], 
        relby_parameters['chi_2z'], relby_parameters['inclination'], relby_parameters['phase'],
        waveform_arguments['reference_frequency'])

    return relby_parameters


def get_binned_detector_response(ifos, parameters, fs, waveform_arguments, mode_array):

    parameters['mass_1'], parameters['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        parameters['chirp_mass'], parameters['mass_ratio'])
    
    Cplus, Ccross, hL = xphm.compute_C_prefactors(parameters, fs, waveform_arguments, np.array(mode_array), fs)

    Fplus = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'], 
        parameters['geocent_time'], parameters['psi'], 'plus') for ifo in ifos])
    Fcross = np.array([ifo.antenna_response(parameters['ra'], parameters['dec'], 
        parameters['geocent_time'], parameters['psi'], 'cross') for ifo in ifos])

    dt_geocent = np.array([ifo.time_delay_from_geocenter(
        parameters['ra'], parameters['dec'], parameters['geocent_time']) for ifo in ifos])
    ## this is basically adding time delay from geocent and duration of the signal
    dt_total = dt_geocent+(parameters['geocent_time']-ifos[0].start_time)

    Fplus = Fplus[:, np.newaxis, np.newaxis]
    Fcross = Fcross[:, np.newaxis, np.newaxis]

    Cplus_projected = Fplus*Cplus
    Ccross_projected = Fcross*Ccross

    Ctotal = Cplus_projected+Ccross_projected
    num_ifos = len(ifos)
    num_modes = len(mode_array)

    exp_timeshift = np.exp(-1j*2*np.pi*fs*dt_total[:, np.newaxis])
    if hL is not None:
        hL = exp_timeshift[:, np.newaxis, :]*hL[np.newaxis, :, :]

    return hL, Ctotal 

def lal_f_max(parameters, reference_frequency=50.):
    spin1 = np.sqrt(parameters['chi_1x']**2+parameters['chi_1y']**2+parameters['chi_1z']**2)
    spin2 = np.sqrt(parameters['chi_2x']**2+parameters['chi_2y']**2+parameters['chi_2z']**2)
    mtotal = parameters['mass_1']+parameters['mass_2']

    chi_eff = (parameters['mass_1']*spin1+parameters['mass_2']*spin2)/mtotal
    msun_sec = 4.925491025543575903411922162094833998e-6
    if chi_eff>0.99:
        return 0.33/(mtotal*msun_sec)
    else:
        return 0.3/(mtotal*msun_sec)

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
    def __init__(self, interferometers, fiducial_model, test_model, priors,
                 fid_fmax, mode_array, waveform_arguments, total_err, grid_choice,
                 simulation=True, delete_strains = False):
        
        """
        Initialization of the class.

        ARGS:
        -----
        - interferometers: list of interferometers that contain the signal
        - fiducial_model: dictionary with the fiducial model parameters to be used
                 for the analysis
        - test_model: dictionary of the test model parameters to be used to make bins. This model will only be used to make 
                        bins not for calculating likelihood later
        - priors: the priors used for the different parameters
        - total_err: total error on the rb likelihood compared to true likelihood
        - delete_strains: if True, deletes strain data after computing summary parameters to free up memory
        """
        
        super(RelativeBinningHOM, self).__init__(dict())
        self.interferometers = interferometers
        self.priors = priors
        self.fid_fmax = np.min([fid_fmax, self.interferometers[0].frequency_array[-1]])
        print('fmax of the analysis', self.fid_fmax)
        self.mask = np.where((self.interferometers[0].frequency_array >= waveform_arguments['minimum_frequency']) &
                             (self.interferometers[0].frequency_array<=self.fid_fmax))[0]
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
        self.h0, self.hC0 = get_binned_detector_response(
            self.interferometers, self.fiducial_model, self.fs,
            self.waveform_arguments, self.mode_array)

        self.test_model = relby_conversion(test_model, self.waveform_arguments)
        self.t0, self.tC0 = get_binned_detector_response(
            self.interferometers, self.test_model, self.fs,
            self.waveform_arguments, self.mode_array)
        self.data = np.array(
            [ifo.frequency_domain_strain[self.mask] for ifo in self.interferometers])
        self.psd_array = np.array(
            [ifo.power_spectral_density_array.copy()[self.mask] for ifo in self.interferometers])
        
        self.grid_choice = grid_choice
        self.noise_log_l = self.noise_log_likelihood()
        self.binning_info = self.setup_bins()
        print('length of rb bins', len(self.binning_info[0]))

        self.parameters = {}
        print('length of the adapted grid', len(self.binning_info[0]))
        print('length of the originl grid', len(self.fs))
        print('length of the grid proposed by bilby', len(self.interferometers[0].frequency_array))
        
        if delete_strains:
            print('Deleting full frequency data to release memory')
            del self.data
            del self.psd_array
            del self.h0
            del self.hC0
            del self.t0
            del self.tC0
            for ifo in self.interferometers:
                del ifo.strain_data
            del self.mask

    def noise_log_likelihood(self):
        """
        Fuction computing the noise log likelihood. 
        This is the same as for the usual runs
        """
        log_l = 0.

        for ifo in self.interferometers:
            ### note that noise log likelihood is evaluated on full freq grid 
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(
                ifo.frequency_domain_strain[mask], ifo.frequency_domain_strain[mask],
                ifo.power_spectral_density_array[mask], self.duration) / 2.

        return float(np.real(log_l))

    def log_likelihood_ratio(self):
       
        fbin, f_bin_ind, h0_fbin, hC0_fbin, A0, A1, B0, B1 = self.binning_info
        p = relby_conversion(self.parameters, self.waveform_arguments)

        h, hC = get_binned_detector_response(
            self.interferometers, p, fbin, self.waveform_arguments, mode_array = self.mode_array)
        
        if h is not None:
            r = (h/h0_fbin)
            s = (hC/hC0_fbin)
            r0, r1 = self.compute_bin_coefficients(fbin, r)
            C0, C1 = self.compute_bin_coefficients(fbin, s)
            Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
            return Zdh - (Zhh/2.)
        else:
            return -np.inf


    def log_likelihood(self):
        log_likelihood = self.log_likelihood_ratio() + self.noise_log_l
        return log_likelihood


    def compute_summary_data(self, fbin, Nbin, fbin_ind, fm):
        """
        Computes the summary data used in mode-by-mode relative binning (Scheme 1).
        fbin: array of frequencies of the bin edges (Hz)
        Nbin: number of bins
        fbin_ind: array of indices of bin edges in frequency array f
        fm: frequencies of the center of the bins (Hz)
        """
        ndet = len(self.interferometers)
        Nmode = self.h0.shape[1]
        T = self.duration
        f = self.fs ## full frequency array 

        A0 = 4./T*np.array([[[np.sum(
            self.data[i, fbin_ind[b]:fbin_ind[b+1]] * np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]) *
            np.conj(self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]]) / self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]])
                              for b in range(Nbin)]
                             for l in range(Nmode)]
                            for i in range(ndet)])
        A1 = 4./T*np.array([[[np.sum(
            self.data[i, fbin_ind[b]:fbin_ind[b+1]] * np.conj(self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]]) *
            np.conj(self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]]) / self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]] *
            (f[fbin_ind[b]:fbin_ind[b+1]]-fm[b]))
                              for b in range(Nbin)]
                             for l in range(Nmode)]
                            for i in range(ndet)])
        B0 = 4./T*np.array([[[[np.sum(
            self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]] * self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]] *
            np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]]) *
            np.conj(self.hC0[i, ll, fbin_ind[b]:fbin_ind[b+1]]) / self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]])
                               for b in range(Nbin)]
                              for l in range(Nmode)]
                             for ll in range(Nmode)]
                            for i in range(ndet)])
        B1 = 4./T*np.array([[[[np.sum(
            self.h0[i, l, fbin_ind[b]:fbin_ind[b+1]] * self.hC0[i, l, fbin_ind[b]:fbin_ind[b+1]] *
            np.conj(self.h0[i, ll, fbin_ind[b]:fbin_ind[b+1]]) *
            np.conj(self.hC0[i, ll, fbin_ind[b]:fbin_ind[b+1]]) / self.psd_array[i, fbin_ind[b]:fbin_ind[b+1]] *
            (f[fbin_ind[b]:fbin_ind[b+1]]-fm[b]))
                               for b in range(Nbin)]
                              for l in range(Nmode)]
                             for ll in range(Nmode)]
                            for i in range(ndet)])
    
        return A0, A1, B0, B1

    def setup_bins(self):
        """
        Compute the binning data for relative binning. This only needs to be done once
        for a particular fiducial model. The results are used in computing the likelihood
        with relative binning for a given model.
        """

        fbin, Nbin, fbin_ind, fm = self.get_binning_data(self.N_initial, self.quad_sum)
        
        self.h0_fbin = self.h0[:, :, fbin_ind]
        self.hC0_fbin = self.hC0[:, :, fbin_ind]

        A0, A1, B0, B1 = self.compute_summary_data(fbin, Nbin, fbin_ind, fm)
        return fbin, fbin_ind, self.h0_fbin, self.hC0_fbin, A0, A1, B0, B1, 

    def get_binning_data(self, N0, quad_sum, i = 0):
        """
        Computes the binning data needed to do mode by mode relative binning with scheme 2
        N0: initial target number of bins
        """
        if self.grid_choice=='bisection':
            fbin, Nbin, fbin_ind, fm = self.get_binning_data_bisect_iteration(N0)
        else:
            fbin, Nbin, fbin_ind, fm = self.get_step_bin_grid(N0, quad_sum = quad_sum)

        print('bin propagation and i ', Nbin, N0, i)
        if i==40:
            print('breakign because iterations reached', i)
            return fbin, Nbin, fbin_ind, fm


        if Nbin==N0:
            if Nbin<40:
                print('Too few bids setting quad sum to False')
                return self.get_binning_data(Nbin, quad_sum = False, i = i+1)
            else:
                return fbin, Nbin, fbin_ind, fm

        else:
            return self.get_binning_data(Nbin, quad_sum = quad_sum, i = i+1)


    def get_binning_data_bisect_iteration(self, N):
        """
        Helper function for get_binning_data_bisect_2. This function does
        bisection checks trying to achieve the target error E in N bins.
        It may require more bins or fewer bins. If so, this function is called again.
        N: target number of bins
        """
        print('Using bisection grid to make bins')
        fmin, fmin_ind = self.nearest_freq(self.fs, self.fs[0])
        fmax, fmax_ind = self.nearest_freq(self.fs, self.fs[-1])
        max_bisections = int(np.floor(np.log2(fmax_ind-fmin_ind)))
        Emax = self.total_err/np.sqrt(N)
        print('Emax of the iteration, divide by sqrt N', Emax)
        fbin_nomin, Nbin, fbin_ind_nomin = self.bisect_bin_search(
            f_lo = fmin, f_hi = fmax, depth = 0, max_depth = max_bisections, Emax = Emax) 
        
        fbin = np.append(fmin, fbin_nomin)
        fbin_ind = np.append(fmin_ind, fbin_ind_nomin)
        fm = (fbin[1:] + fbin[:-1])/2
        return fbin, Nbin, fbin_ind, fm

    def bisect_bin_search(self, f_lo, f_hi, depth, max_depth, Emax):
        """
        Helper function for get_binning_data_bisect_2_iteration.
        This recursive function does the bisection until either the target
        is achieved or the maximum bisection depth is reached.
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
        if bin_error < Emax:
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

    def compute_bin_error(self, f_lo, f_hi):
        """
        Computes the error contribution to the log likelihood of
        mode-by-mode relative binning scheme 2 for one bin that
        begins at the frequency in the data object closest to f_lo
        and ends at the frequency in the data object closest to f_hi.
        f_lo: target lower frequency for the bin, this exact frequency may not be in the frequency array
        f_hi: target upper frequency for the bin, this exact frequency may not be in the frequency array
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
        
        r0, r1 = self.compute_bin_coefficients(fbin, r)
        C0, C1 = self.compute_bin_coefficients(fbin, s)

        Zdh, Zhh = self.compute_overlaps(r0, r1, C0, C1, A0, A1, B0, B1)
        rb_lnL = Zdh - 1/2*Zhh
        
        t_allmodes = np.sum(self.tC0*self.t0, axis = 1)
        
        exact_lnL = np.sum(
            [inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]],
                      self.data[i, fbin_ind[0]:fbin_ind[1]],
                      self.psd_array[i, fbin_ind[0]:fbin_ind[1]], self.duration) -
             1/2*inn_prod(t_allmodes[i, fbin_ind[0]:fbin_ind[1]],
                          t_allmodes[i, fbin_ind[0]:fbin_ind[1]],
                          self.psd_array[i, fbin_ind[0]:fbin_ind[1]], self.duration)
             for i in range(ndet)])
       
        return np.abs(rb_lnL - exact_lnL)

    def nearest_freq(self, f, target_f):
        """
        Returns the frequency in f closest to target_f.
        f: frequency array
        target_f: target frequency
        """
        f_ind = np.searchsorted(f, target_f)
        f_near = f[f_ind]
        return f_near, f_ind

    def compute_overlaps(self, r0, r1, C0, C1, A0, A1, B0, B1):
        """
        Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf)
        used in advanced mode by mode relative binning (Scheme 1).
        r0, r1: linear coefficients for the ratio of time-dependent L-frame modes
                over fiducial time-dependent L-frame modes, indices: detector, mode, frequency
        C0, C1: linear coefficients for the coefficient of the time-dependent
                L-frame modes, indices: detector, mode, frequency
        A0, A1, B0, B1: summary data for mode-by mode relative binning, A
                        ndices: detector, mode, frequency, B indices: detector, mode, mode, frequency
        """
        
        Zdh = np.real(np.sum(
            A0*np.conj(r0)*np.conj(C0) 
            + A1*( (np.conj(r0)*np.conj(C1)) + (np.conj(r1)*np.conj(C0)) ) ))
        
        Zhh = np.real(np.sum(
            B0*r0[:, np.newaxis, :, :] * C0[:, np.newaxis, :, :] *
            np.conj(r0[:, :, np.newaxis, :]) * np.conj(C0[:, :, np.newaxis, :])
            + B1*( r0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])
                  * (np.conj(C0[:, :, np.newaxis, :]) *C1[:, np.newaxis, :, :] +
                     C0[:, np.newaxis, :, :]*np.conj(C1[:, :, np.newaxis, :])) +
                  C0[:, np.newaxis, :, :]*np.conj(C0[:, :, np.newaxis, :])*
                  (np.conj(r0[:, :, np.newaxis, :])*r1[:, np.newaxis, :, :] +
                   r0[:, np.newaxis, :, :]*np.conj(r1[:, :, np.newaxis, :])) ) ))
        
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
            j = 1 # steps in which the bins will increase
            while (i+j)<len(self.fs):
                flo = self.fs[i]
                fhi = self.fs[i+j]
                add_bins = i+j
                bin_error = self.compute_bin_error(flo, fhi)
                j+=1

                if bin_error>Emax:
                    i = i+j-1
                    output_step_bins.append(self.fs[add_bins])
                    output_step_bins_index.append(add_bins)
                    break
            if (i+j)==len(self.fs):
                print('Breaking because we reached the end')
                break

        output_step_bins = np.unique(np.array(output_step_bins))
        Nbin = len(output_step_bins)
        output_step_bins_index = np.unique(np.array(output_step_bins_index))
        fm = (output_step_bins[1:] + output_step_bins[:-1])/2
        return (np.unique(np.array(output_step_bins)),
                len(output_step_bins)-1,
                np.unique(np.array(output_step_bins_index)),
                fm)


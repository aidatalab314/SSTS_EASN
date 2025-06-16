import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

sensor_step = 0.000136533

def cluster(sig, minmax):

    x = []
    y = []
    for i in range(len(sig)):
        if sig[i] > minmax[0] and (not minmax[1] or sig[i] < minmax[1]):
            x.append(i)
            y.append(sig[i])

    return x, y 


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def plot_specgram(data, title='', x_label='', y_label='', fig_size=None):

    n = len(data)
    fs = 1 / sensor_step #np.max(np.fft.fft(data, n)) * 2

    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    pxx,  freq, t, cax = plt.specgram(data, Fs = fs, mode = 'magnitude')
    fig.colorbar(cax).set_label('Amplitude [dB]')
    plt.show()





def sig_plot(x, max, target_dir, prefix, xl = None, yl = None):

    flip = max < 0
    max = abs(max)
    n = len(x)
    final_ts = sensor_step * max * n 

    freq = sensor_step * np.arange(n)
    L = np.arange(1, np.floor(max * n), dtype = 'int')
    if flip:
        L = np.flip(L)

    plt.figure(figsize=(7, 4)) 
    plt.plot(freq[L], x[L].real)
    if xl and yl:
        plt.xlabel(xl)
        plt.ylabel(yl)

    plt.xlim(-0.1 * final_ts, 1.1 * final_ts)
    #plt.ylim(-4, 4)
    #plt.ylim(-1 * plot, plot)

    plt.title('{} {}'.format(prefix, target_dir))
    plt.show()

def fft_denoiser(x, minmax, target_dir = None, plot = None, cls = None):
    """
    Fast Fourier Transform (FFT) High-Pass Denoiser
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    minmax : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """

    n = len(x)
    
    trans = np.fft.fft(x, n)
    fx = np.array(trans)

    # realnum = np.real(trans)
    # comnum = np.imag(trans)
    # mag = np.sqrt(realnum ** 2 + comnum ** 2) + 1e-5

    # #   Spectral Residual
    # spectral = np.exp(np.log(mag) - average_filter(np.log(mag)))
    # spectral2 = mag - average_filter(mag)

    #   Compute power spectral density
    #   Squared magnitude of each fft coefficient
    PSD = trans * np.conj(trans) / n


    #   BPF
    _mask = minmax[0] < PSD 

    if minmax[1]:
        _mask2 = PSD < minmax[1]
        _mask = [a and b for a, b in zip(_mask, _mask2)]

    trans = _mask * trans

    masked = np.array(trans)
    #   Inverse fourier transform
    clean_data = np.fft.ifft(trans)
    clean_data = clean_data.real
    
    # sm_trans = trans
    # sm_trans.real = sm_trans.real * spectral / mag
    # sm_trans.imag = sm_trans.imag * spectral / mag

    # saliency_map = np.fft.ifft(sm_trans)

    if plot:
        sig_plot(x, 1, target_dir, 'RAW', 'Time', 'Amplitude')
        sig_plot(scipy.fftpack.fft(x), 1, target_dir, 'FFT_scipy')
        sig_plot(fx, -0.5, target_dir, 'FFT', 'Frequency', 'Amplitude')
        #sig_plot(mag, -0.5, target_dir, 'MAG')
        #sig_plot(spectral, 1, target_dir, 'SR')
        #sig_plot(comnum, -0.5, target_dir, 'COM')
        #sig_plot(comnum ** 2, -0.5, target_dir, 'COM2')
        #sig_plot(realnum, -0.5, target_dir, 'REAL')
        #sig_plot(realnum ** 2, -0.5, target_dir, 'REAL2')
        #sig_plot(spectral2, -0.5, target_dir, 'SR')
        #sig_plot(saliency_map, 1, target_dir, 'SM')
        sig_plot(masked, -0.5, target_dir, 'MASK_FFT', 'Frequency', 'Amplitude')
        sig_plot(PSD, -0.5, target_dir, 'PSD', 'Frequency', 'Power Density')
        sig_plot(clean_data, 1, target_dir, 'IFFT', 'Time', 'Amplitude')

        #thingy = np.array([i if i != 0 else 100 for i in x - clean_data])

        #sig_plot(thingy,    1,      target_dir, 'amogus')

    if cls:
        return cluster(PSD.real, minmax)
    
    return clean_data
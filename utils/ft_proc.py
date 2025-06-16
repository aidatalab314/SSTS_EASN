import numpy as np

class ft_ext(object):
    def __init__(self, data, no_elements):
        self.data = data
        self.no_elements = no_elements

    #   Generate mean data based on raw data.
    def mean(self):
        print('Generating mean data.')
        X = np.zeros((self.data.shape[0], self.data.shape[1]))

        #   Slide window of length no_elements over every list in sampled data.
        for i in range(self.data.shape[1] - self.no_elements + 1):

            #   Obtain the mean of every window.
            X[:,i] = np.mean(self.data[:, i:i + self.no_elements], axis = 1)

        return X.astype(np.float16)
    
    #   Generate absolute mean data based on raw data.
    def abs_mean(self):
        print('Generating absolute mean data.')
        X = np.zeros((self.data.shape[0], self.data.shape[1]))

        self.data = np.abs(self.data)

        #   Slide window of length no_elements over every list in sampled data.
        for i in range(self.data.shape[1] - self.no_elements + 1):

            #   Obtain the mean of every window.
            X[:,i] = np.mean(self.data[:, i:i + self.no_elements], axis = 1)

        return X.astype(np.float16)

    #   Generate variance data based on raw data.
    def variance(self):
        print('Generating variance data.')
        X = np.zeros((self.data.shape[0], self.data.shape[1]))

        #   Slide window of length no_elements over every list in sampled data.
        for i in range(self.data.shape[1] - self.no_elements + 1):

            #   Obtain the rms of every window.
            X[:,i] = np.var(self.data[:, i:i + self.no_elements], axis = 1)

        return X.astype(np.float16)

    #   Generate root mean square data based on raw data.
    def rms(self):
        print('Generating RMS data.')
        X = np.zeros((self.data.shape[0], self.data.shape[1]))

        #   Slide window of length no_elements over every list in sampled data.
        for i in range(self.data.shape[1] - self.no_elements + 1):

            #   Obtain the rms of every window.
            X[:,i] = np.sqrt(np.mean(np.square(self.data[:, i:i + self.no_elements]), axis = 1))

        return X.astype(np.float16)

    #   Generaet absolute change data based on raw data.
    def abs_change(self):
        X = np.zeros((self.data.shape[0], self.data.shape[1]))
        X[:,0] = self.data[:,0]

        for i in range(1, self.data.shape[1]):
            X[:,i] = self.data[:,i] - self.data[:,i - 1]
        
        return X.astype(np.float16)

def to_2D(data, size):
    X = np.zeros((data.shape[0], size, size))

    for i in range(data.shape[0]):
        X[i] = (data[i, :].reshape(size, size))
    
    return X.astype(np.float16)
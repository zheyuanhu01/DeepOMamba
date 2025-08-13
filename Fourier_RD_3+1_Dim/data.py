import abc
import numpy as np
import torch
from sklearn import gaussian_process as gp
from scipy import interpolate
from scipy.integrate import solve_ivp

class Data:
    '''Standard data format. 
    '''
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.__device = None
        self.__dtype = None
    
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype
    
    @device.setter    
    def device(self, d):
        if d == 'cpu':
            self.__to_cpu()
        elif d == 'gpu':
            self.__to_gpu()
        else:
            raise ValueError
        self.__device = d
    
    @dtype.setter     
    def dtype(self, d):
        if d == 'float':
            self.__to_float()
        elif d == 'double':
            self.__to_double()
        else:
            raise ValueError
        self.__dtype = d
    
    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')
    
    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64
    
    @property
    def dim(self):
        if isinstance(self.X_train, np.ndarray):
            return self.X_train.shape[-1]
        elif isinstance(self.X_train, torch.Tensor):
            return self.X_train.size(-1)
    
    @property
    def K(self):
        if isinstance(self.y_train, np.ndarray):
            return self.y_train.shape[-1]
        elif isinstance(self.y_train, torch.Tensor):
            return self.y_train.size(-1)
    
    @property
    def X_train_np(self):
        return Data.to_np(self.X_train)
    
    @property
    def y_train_np(self):
        return Data.to_np(self.y_train)
    
    @property
    def X_test_np(self):
        return Data.to_np(self.X_test)
    
    @property
    def y_test_np(self):
        return Data.to_np(self.y_test)
    
    @staticmethod      
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        elif isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        else:
            raise ValueError
    
    def __to_cpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.DoubleTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cpu())
    
    def __to_gpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.cuda.DoubleTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cuda())
    
    def __to_float(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).float())
    
    def __to_double(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).double())

class ODEData(Data, abc.ABC):
    '''dy/dt=g(y,u,t), y(0)=s0, 0<=t<=T.
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(ODEData, self).__init__()
        self.T = T
        self.s0 = s0
        self.sensor_in = sensor_in
        self.sensor_out = sensor_out
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    @abc.abstractmethod
    def g(self, y, u, t):
        pass
        
    def __init_data(self):
        features = 1000 * self.T
        train = self.__gaussian_process(self.train_num, features)
        test = self.__gaussian_process(self.test_num, features)
        self.X_train = self.__sense(train).reshape([-1, self.sensor_in, 1])
        self.y_train = self.__solve(train).reshape([-1, self.sensor_out, 1])
        self.X_test = self.__sense(test).reshape([-1, self.sensor_in, 1])
        self.y_test = self.__solve(test).reshape([-1, self.sensor_out, 1])
    
    def __gaussian_process(self, num, features):
        x = np.linspace(0, self.T, num=features)[:, None]
        K = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(K + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()
    
    def __sense(self, gps):
        x = np.linspace(0, self.T, num=gps.shape[1])
        res = map(
            lambda y: interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True
            )(np.linspace(0, self.T, num=self.sensor_in)),
            gps)
        return np.vstack(list(res))
    
    def __solve(self, gps):
        x = np.linspace(0, self.T, num=gps.shape[1])
        interval = np.linspace(0, self.T, num=self.sensor_out) if self.sensor_out > 1 else [self.T]
        def solve(y):
            u = interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True)
            return solve_ivp(lambda t, y: self.g(y, u(t), t), [0, self.T], self.s0, 'RK45', interval, max_step=0.05).y[0]
        return np.vstack(list(map(solve, gps)))

class AntideData(ODEData):
    '''Data for learning the antiderivative operator.
    g(y,u,t)=u.
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(AntideData, self).__init__(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    
    def g(self, y, u, t):
        return u

class AntideAntideData(ODEData):
    '''Data for learning the gravity pendulum.
    g(y,u,t)=[y[1], u].
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(AntideAntideData, self).__init__(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
        
    def g(self, y, u, t):
        return [y[1], u]
  
if __name__ == "__main__":
    device = 'gpu' # 'cpu' or 'gpu'
    # data
    T = 1
    s0 = [0]
    sensor_in = 1000
    sensor_out = 1000
    length_scale = 0.2
    train_num = 10
    test_num = 10

    np.random.seed(0)
    data = AntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    X = data.X_train
    y = data.y_train

    np.random.seed(0)
    s0 = [0, 0]
    data = AntideAntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    z = data.y_train

    dydt_pred = (y[:, 2:] - y[:, :-2]) * 500
    dydt_true = X[:, 1:-1]
    err = np.sum((dydt_pred - dydt_true) ** 2) / np.sum(dydt_true ** 2)
    print(np.sqrt(err))

    dydt_pred = (z[:, 2:] - z[:, :-2]) * 500
    dydt_true = y[:, 1:-1]
    err = np.sum((dydt_pred - dydt_true) ** 2) / np.sum(dydt_true ** 2)
    print(np.sqrt(err))

    pass
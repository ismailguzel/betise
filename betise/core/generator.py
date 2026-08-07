import numpy as np
import pandas as pd
import random
from numpy.polynomial import Polynomial
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from statsmodels.tsa.seasonal import STL,MSTL
from betise.utils.arfima_simulator import ARFIMA_sim

class TimeSeriesGenerator:
    def __init__(self, length=None):
        self.length = length if length is not None else 400
        self.stationary_base_distributions = ['ar', 'ma', 'arma','white_noise']
        self.volatile_base_distributions = ['arch', 'garch', 'egarch', 'aparch']
        self.stochastic_base_distributions = ['ari', 'ima', 'arima']
        self.fractional_base_distributions = ['arfima']
        self.characteristics = {'deterministic_trend_linear' : self.generate_deterministic_trend_linear,
        'deterministic_trend_cubic': self.generate_deterministic_trend_cubic,
        'deterministic_trend_quadratic': self.generate_deterministic_trend_quadratic,
        'deterministic_trend_exponential': self.generate_deterministic_trend_exponential,
        'deterministic_trend_damped': self.generate_deterministic_trend_damped,
        'stochastic_trend': self.generate_stochastic_trend,
        'fractional_process': self.generate_fractional_process,
        'single_seasonality': self.generate_single_seasonality,
        'multiple_seasonality': self.generate_multiple_seasonality,
        'sarima_seasonality': self.generate_sarima_series,
        'sarma_seasonality': self.generate_sarma_series,
        'single_point_anomaly' : self.generate_point_anomaly,
        'multiple_point_anomalies': self.generate_point_anomalies,
        'collective_anomalies': self.generate_collective_anomalies,
        'contextual_anomalies': self.generate_contextual_anomalies}
        self.structural_breaks = {'mean_shift': self.generate_mean_shift,
        'variance_shift': self.generate_variance_shift,
        'trend_shift': self.generate_trend_shift}

    #HELPER FUNCTIONS

    # Check if AR parameters lead to stationarity
    def is_stationary(self, ar_params):
        ar_poly = np.r_[1, -ar_params]
        roots = Polynomial(ar_poly).roots()
        return np.all(np.abs(roots) > 1)

    # Check if MA parameters lead to invertibility
    def is_invertible(self, ma_params):
        ma_poly = np.r_[1, ma_params]
        roots = Polynomial(ma_poly).roots()
        return np.all(np.abs(roots) > 1)

    def generate_nonzero_coefs(self, order, low, high, exclusion_lower, exclusion_upper):
        coefs = []
        while len(coefs) < order:
            val = np.random.uniform(low, high)
            if abs(val) >= exclusion_lower and abs(val) <= exclusion_upper:
                coefs.append(val)
        return np.array(coefs)
        

    #BASE DISTRIBUTIONS STATIONARY

    def generate_ar_params(self, order_range=(1, 5), coef_range=(-0.9, 0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = np.random.uniform(coef_range[0], coef_range[1], order)
            ar = np.r_[1, -coefs]
            ma = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isstationary:
                break
        return order, coefs

    def generate_ar_series(self, length, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        order,coefs = self.generate_ar_params()
        info = {'type': 'base_series', 'subtype': 'AR', 'ar_order': order, 'ar_coefs': coefs} 
        ar = np.r_[1, -np.array(coefs)]  # leading 1 and negate the coefficients
        ma = np.r_[1]  # MA coefficients are just [1] for a pure AR process
        ar_process = ArmaProcess(ar, ma)
        series = ar_process.generate_sample(nsample=length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info
            
    def generate_ma_params(self, order_range=(1, 5), coef_range=(-0.9, 0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = np.random.uniform(coef_range[0], coef_range[1], order)
            ma = np.r_[1, coefs]
            ar = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isinvertible:
                break
        return order, coefs

    def generate_ma_series(self, length, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        order,coefs = self.generate_ma_params()
        info = {'type': 'base_series','subtype': 'MA', 'ma_order': order, 'ma_coefs': coefs}
        ar = np.r_[1]  # AR coefficients are just [1] for a pure MA process
        ma = np.r_[1, np.array(coefs)]  # leading 1 for the MA coefficients
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_arma_params(self, order_range=(1, 5), coef_range=(-0.9, 0.9)):
        while True:
            ar_order = np.random.randint(order_range[0], order_range[1] + 1)
            ma_order = np.random.randint(order_range[0], order_range[1] + 1)
            ar_coefs = np.random.uniform(coef_range[0], coef_range[1], ar_order)
            ma_coefs = np.random.uniform(coef_range[0], coef_range[1], ma_order)
            ma = np.r_[1, ma_coefs]
            ar = np.r_[1, -ar_coefs]
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isinvertible and arma_process.isstationary:
                break
        return ar_order, ma_order, ar_coefs, ma_coefs

    def generate_arma_series(self, length, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        ar_order,ma_order,ar_coefs,ma_coefs = self.generate_arma_params()
        info = {'type': 'base_series', 'subtype': 'ARMA', 'ar_order': ar_order, 'ar_coefs': ar_coefs, 'ma_order': ma_order, 'ma_coefs': ma_coefs}
        ar = np.r_[1, -np.array(ar_coefs)]
        ma = np.r_[1, np.array(ma_coefs)]
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_white_noise(self, length, noise_std = None):
        info = {'type': 'base_series','subtype': 'white_noise'}
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        series = np.random.normal(0, 1, length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_arima_params(self, order_range=(1, 3), coef_range = (-0.9,0.9)):
        while True:
            p = np.random.randint(order_range[0], order_range[1] + 1)
            q = np.random.randint(order_range[0], order_range[1] + 1)

            ar_coefs = self.generate_nonzero_coefs(p, coef_range[0], coef_range[1], exclusion_lower=0.2, exclusion_upper=0.8)
            ma_coefs = self.generate_nonzero_coefs(q, coef_range[0], coef_range[1], exclusion_lower=0.2, exclusion_upper=0.8)

            ar = np.r_[1, -ar_coefs]
            ma = np.r_[1, ma_coefs]

            arma_process = ArmaProcess(ar, ma)
            if arma_process.isstationary and arma_process.isinvertible:
                break

        return p, q, ar_coefs, ma_coefs

    def generate_arima_series(self, length, d= 1, const=False, drift=None, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        p, q, ar_coefs, ma_coefs = self.generate_arima_params()

        ar = np.r_[1, -ar_coefs]
        ma = np.r_[1, ma_coefs]

        if d == 1: 
            unit_root_label = "1_unit_root"
        elif d == 2:
            unit_root_label = "2_unit_root"

        info = {'type': 'trend', 'subtype' : 'stochastic_ARIMA', 'unit_root': unit_root_label,
                 'ar_order': p, 'ar_coefs': ar_coefs, 'ma_order': q, 'ma_coefs': ma_coefs, 'diff': d}
        arma_process = ArmaProcess(ar, ma)
        arma_sample = arma_process.generate_sample(nsample=length)

        # Integrate (difference 'd' times)
        series = arma_sample
        for _ in range(d):
            series = np.cumsum(series)
        if const:
            if drift is None:
                drift = np.random.uniform(0.01, 0.08)
            series += drift * np.arange(length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_ari_params(self, order_range=(1, 3), coef_range = (-0.9,0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = self.generate_nonzero_coefs(order, coef_range[0], coef_range[1], exclusion_lower = 0.3, exclusion_upper = 0.6)
            ar = np.r_[1, -coefs]
            ma = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isstationary:
                break
        return order, coefs

    def generate_ari_series(self, length, d = 1, const=False, drift=None, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        order, coefs = self.generate_ari_params()
        if d == 1: 
            unit_root_label = "1_unit_root"
        elif d == 2:
            unit_root_label = "2_unit_root"
        info = {'type': 'trend', 'subtype' : 'stochastic_ARI', 'unit_root': unit_root_label,
                 'ar_order': order, 'ar_coefs': coefs, 'diff': d}
        ar = np.r_[1, -coefs]
        ma = np.array([1])
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=length)
        for _ in range(d):
            series = np.cumsum(series)
        if const:
            if drift is None:
                drift = np.random.uniform(0.01, 0.08)
            series += drift * np.arange(length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_ima_params(self, order_range=(1, 3), coef_range = (-0.9,0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = self.generate_nonzero_coefs(order, coef_range[0], coef_range[1], exclusion_lower = 0.3, exclusion_upper = 0.6)
            ar = np.array([1])
            ma = np.r_[1, coefs]
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isinvertible:
                break
        return order, coefs

    def generate_ima_series(self, length, d = 1, const=False, drift=None, noise_scale=0.5, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        order, coefs = self.generate_ima_params()
        if d == 1: 
            unit_root_label = "1_unit_root"
        elif d == 2:
            unit_root_label = "2_unit_root"
        info = {'type': 'trend', 'subtype' : 'stochastic_IMA', 'unit_root': unit_root_label, 
                'ma_order': order, 'ma_coefs': coefs, 'diff': d}
        ar = np.array([1])
        ma = np.r_[1, coefs]
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=length)
        for _ in range(d):
            series = np.cumsum(series)
        if const:
            if drift is None:
                drift = np.random.uniform(0.01, 0.8)
            series += drift * np.arange(length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_arfima_params(
        self,
        order_range=(1, 3),
        d_range=(0.25, 0.49),
        coef_range=(-0.9, 0.9)
    ):
        """Generate ARFIMA model parameters.
        
        Returns (p, d, q, ar_coefs, ma_coefs) where:
        - p, q are AR/MA orders
        - d is fractional differencing parameter (0.25 to 0.49 for long memory)
        - ar_coefs, ma_coefs are stationary/invertible coefficients
        """
        while True:
            p = np.random.randint(order_range[0], order_range[1] + 1)
            q = np.random.randint(order_range[0], order_range[1] + 1)

            ar_coefs = self.generate_nonzero_coefs(
                p, coef_range[0], coef_range[1],
                exclusion_lower=0.2, exclusion_upper=0.8
            )

            ma_coefs = self.generate_nonzero_coefs(
                q, coef_range[0], coef_range[1],
                exclusion_lower=0.2, exclusion_upper=0.8
            )

            ar = np.r_[1, -ar_coefs]
            ma = np.r_[1, ma_coefs]
            arma_process = ArmaProcess(ar, ma)

            if arma_process.isstationary and arma_process.isinvertible:
                break

        # Fractional differencing parameter for long memory
        d = np.random.uniform(d_range[0], d_range[1])

        return p, d, q, ar_coefs, ma_coefs

    def generate_arfima_series(
        self,
        length,
        d_range=(0.25, 0.49),
        noise_std=None,
        alpha=0,
        numseas=100
    ):
        """Generate ARFIMA(p, d, q) series with long memory.
        
        Parameters
        ----------
        length : int
            Series length
        d_range : tuple
            Range for fractional differencing parameter
        noise_std : float, optional
            Innovation standard deviation
        alpha : float
            Additive series constant, i.e. the process mean for a stationary
            series (default: 0)
        numseas : int
            Number of seasoning (burn-in) samples generated and discarded
            before the series is recorded (default: 100)

        Returns
        -------
        series : np.ndarray
            Generated ARFIMA series
        info : dict
            Metadata including d, stationarity, long_memory flags
        """
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        
        p, d, q, ar_coefs, ma_coefs = self.generate_arfima_params(d_range=d_range)

        info = {
            'type': 'base_series',
            'subtype': 'ARFIMA',
            'p': p,                # ARFIMA p order
            'd': d,                # Fractional differencing parameter
            'q': q,                # ARFIMA q order
            'ar_order': p,         # Alias for consistency
            'ar_coefs': ar_coefs,
            'ma_order': q,         # Alias for consistency
            'ma_coefs': ma_coefs,
            'diff': d,             # Alias for consistency with other processes
            'stationary': d < 0.5,
            'fractionally_integrated': True,
            'long_memory': 0 < d < 0.5,
            'alpha': alpha,
            'sigma': noise_std,
            'numseas': numseas
        }

        series = ARFIMA_sim(
            p_coeffs=ar_coefs,
            q_coeffs=ma_coefs,
            d=d,
            slen=length,
            alpha=alpha,
            sigma=noise_std,
            numseas=numseas
        )

        return series, info

    def generate_fractional_process(
        self,
        kind=None,
        d_range=(0.25, 0.49),
        noise_std=None,
        alpha=0,
        numseas=100
    ):
        """Generate fractionally integrated process (ARFIMA).
        
        Parameters
        ----------
        kind : str, optional
            Type of fractional process ('arfima'). If None, random choice.
        d_range : tuple
            Range for fractional differencing parameter
        noise_std : float, optional
            Innovation standard deviation
        alpha : float
            Additive series constant
        numseas : int
            Number of seasoning (burn-in) samples discarded before recording

        Returns
        -------
        df : pd.DataFrame
            DataFrame with time, data, and classification columns
        info : dict
            Metadata dictionary
        """
        if kind is None:
            kind = np.random.choice(self.fractional_base_distributions)

        if kind == 'arfima':
            series, info = self.generate_arfima_series(
                length=self.length,
                d_range=d_range,
                noise_std=noise_std,
                alpha=alpha,
                numseas=numseas
            )
        else:
            raise ValueError(
                f"Invalid fractional process '{kind}'. "
                f"Choose from {self.fractional_base_distributions}."
            )

        d = float(info['diff'])

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': np.full(self.length, int(d < 0.5), dtype=int),
            'seasonal': np.zeros(self.length, dtype=int),
        })

        return df, info
    
    def generate_arch_series(self, length, alpha_range=(0.5, 0.9), omega_range=(0.1, 0.3), cumulative=False, scale_factor=1):
        alpha = np.random.uniform(*alpha_range)
        omega = np.random.uniform(*omega_range)
        
        am = arch_model(None, vol='ARCH', p=1, mean='Zero')
        sim = am.simulate([omega, alpha], nobs=length)
        
        series = sim['data'].values * scale_factor
        info = {'type': 'volatility', 'subtype': 'ARCH', 'alpha': alpha, 'omega': omega}
        if cumulative:
            series = np.cumsum(series)
    
        return series, info

    def generate_garch_series(self, length, alpha_range=(0.4, 0.6), beta_range=(0.2, 0.5), omega_range=(0.3, 0.6), cumulative=False, scale_factor=1):
        while True:
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            omega = np.random.uniform(*omega_range)
            if alpha + beta < 1:
                break  # Ensure weak stationarity of the variance
    
        am = arch_model(None, vol='GARCH', p=1, q=1, mean='Zero')
        sim = am.simulate([omega, alpha, beta], nobs=length)
        
        series = sim['data'].values * scale_factor
        info = {'type': 'volatility', 'subtype': 'GARCH', 'alpha': alpha, 'beta': beta, 'omega': omega}
        if cumulative:
            series = np.cumsum(series)
    
        return series, info

    def generate_egarch_series(self, length, omega_range=(0.1, 0.3), alpha_range=(0.5, 0.9), beta_range=(0.6, 0.9), theta_range=(-0.5, 0.5), lambda_range=(0.1, 0.5), cumulative=False, scale_factor=1):
        omega = np.random.uniform(*omega_range)
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        theta = np.random.uniform(*theta_range)
        lam = np.random.uniform(*lambda_range)

        am = arch_model(None, vol='EGARCH', p=1, q=1, mean='Zero', dist='normal')
        sim = am.simulate([omega, alpha, beta, theta, lam], nobs=length)

        series = sim['data'].values * scale_factor
        info = {'type': 'volatility', 'subtype': 'EGARCH', 'alpha': alpha, 'beta': beta, 'theta': theta, 'lambda': lam, 'omega': omega}
        if cumulative:
            series = np.cumsum(series)

        return series, info

    def generate_aparch_series(self, length, omega_range=(0.1, 0.3), alpha_range=(0.1, 0.3), beta_range=(0.5, 0.8), gamma_range=(-0.3, 0.3), delta_range=(1.0, 2.0), cumulative=False, scale_factor=1):
        # Stationarity constraint: alpha + beta < 1
        while True:
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            if alpha + beta < 1:
                break
        
        omega = np.random.uniform(*omega_range)
        gamma = np.random.uniform(*gamma_range)
        delta = np.random.uniform(*delta_range)

        from arch import arch_model
        am = arch_model(None, vol='APARCH', p=1, o=1, q=1, mean='Zero', dist='normal')

        sim = am.simulate([omega, alpha, gamma, beta, delta], nobs=length)

        series = sim['data'].values * scale_factor
        info = {'type': 'volatility', 'subtype': 'APARCH', 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'omega': omega}
        if cumulative:
            series = np.cumsum(series)

        return series, info

    def generate_stationary_base_series(self, distribution=None):
        if distribution is None:
            distribution = np.random.choice(self.stationary_base_distributions)
        if distribution == 'white_noise':
            series, info = self.generate_white_noise(self.length)
        elif distribution == 'ar':
            series, info = self.generate_ar_series(self.length)
        elif distribution == 'ma':
            series, info = self.generate_ma_series(self.length)
        elif distribution == 'arma':
            series, info = self.generate_arma_series(self.length)

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': (np.ones(self.length)).astype(int),
            'seasonal': np.zeros(self.length).astype(int),

        })
        return df, info

    #ANOMALIES    

    def generate_point_anomaly(self, df, location=None, scale_factor=1, is_spike=True, is_loc = None):
        series = df['data'].copy()
        n = len(series)
        num_anomalies = 1
    
        # Determine candidate indices based on location
        if location == "beginning":
            candidate_range = np.arange(int(0.1 * n), int(0.3 * n))
        elif location == "middle":
            candidate_range = np.arange(int(0.4 * n), int(0.6 * n))
        elif location == "end":
            candidate_range = np.arange(int(0.7 * n), int(0.9 * n))
        else:
            candidate_range = np.arange(int(0.1 * n), int(0.9 * n))  # Default safe zone
    
        if len(candidate_range) == 0:
            raise ValueError("No valid candidate indices found for the given location.")
    
        # Select point anomaly index
        anomaly_indices = np.random.choice(candidate_range, num_anomalies, replace=False)
    
        # Inject anomaly guaranteed to be dominant
        for idx in anomaly_indices:
            local_std = np.std(series[max(0, idx - int(n*0.5)):min(n, idx + int(n*0.5))])
            global_spike = np.max(np.abs(series - np.mean(series)))
            global_spike_factor = np.random.uniform(1.1,1.3)
            if is_spike:
                magnitude = global_spike_factor * global_spike * scale_factor
            else:
                magnitude = local_std * np.random.uniform(1.5, 2.5) * scale_factor
            direction = np.random.choice([-1, 1])
            series[idx] = np.mean(series) + direction * magnitude
    
        info = {'type': 'anomaly', 'subtype': 'single_point', 'num_anomalies': num_anomalies, 'anomaly_indices': anomaly_indices, 'location': location}
    
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'point_anom_single'] = 1

        if is_loc:
            point_anom_label = np.zeros(n, dtype=int)
            point_anom_label[anomaly_indices] = 1
            df.loc[:, "point_anom_label"] = point_anom_label

        return df, info

    def generate_point_anomalies(self, df, scale_factor=1,is_loc=None):
        series = df['data'].copy()
        n = len(series)

        def compute_point_anomaly_count(length):
            min_anom = 2
            max_anom = min(40, int(length * 0.02))

            if max_anom <= min_anom:
                return min_anom
            return np.random.randint(min_anom, max_anom + 1)
    
        # Determine how many anomalies to inject
        num_anomalies = compute_point_anomaly_count(n)
    
        # Select point anomaly indices
        anomaly_indices = np.random.choice(n, num_anomalies, replace=False)
        anomaly_indices = np.sort(anomaly_indices)
    
        # Compute the max deviation from the mean — natural peak size
        global_spike = np.max(np.abs(series - np.mean(series)))
        for idx in anomaly_indices:
            local_window = series[max(0, idx - int(n*0.5)):min(n, idx + int(n*0.5))]
            local_std = np.std(local_window)
    
            # Choose base magnitude using local std with randomness
            base_mag = local_std * np.random.uniform(2, 3.5)
    
            # Enforce visibility: must be at least 1.1× natural spike
            global_spike_factor = np.random.uniform(0.5,1.2)
            min_visible_mag = global_spike_factor * global_spike
            magnitude = max(base_mag, min_visible_mag) * scale_factor
            
            # Add anomaly
            direction = np.random.choice([-1, 1])
            series[idx] = np.mean(series) + direction * magnitude
    
        info = {'type': 'anomaly', 'subtype': 'multiple_point','num_anomalies': num_anomalies, 'anomaly_indices': anomaly_indices}
    
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'point_anom_multi'] = 1

        if is_loc:
            point_anom_label = np.zeros(n, dtype=int)
            point_anom_label[anomaly_indices] = 1
            df.loc[:, "point_anom_label"] = point_anom_label

        return df, info

    def generate_collective_anomalies(
        self,
        df,
        num_anomalies=1,
        location="middle",
        scale_factor=1,
        anomaly_shapes="rectangular",
        edge_margin=0.05,
        min_distance=0.10,
        max_attempts=1000,
        is_loc = None,
    ):
        series = df["data"].copy()
        original_series = series.copy()
        n = len(series)

        shape_configs = {
            "rectangular": {
                "length_range": (0.05, 0.09),
                "magnitude_range": (1, 1.75),
                "residual_weight": None,
                "method": "add"
            },
            "gaussian": {
                "length_range": (0.09, 0.15),
                "magnitude_range": (1.5, 2.5),
                "residual_weight": 0.1,
                "method": "baseline"
            },
            "triangular": {
                "length_range": (0.09, 0.15),
                "magnitude_range": (1.5, 2.5),
                "residual_weight": 0.15,
                "method": "baseline"
            },
            "ramp": {
                "length_range": (0.05, 0.1),
                "magnitude_range": (1.5, 2.5),
                "residual_weight": 0.15,
                "method": "baseline"
            },
            "decay": {
                "length_range": (0.05, 0.1),
                "magnitude_range": (1.5, 2.5),
                "residual_weight": 0.15,
                "method": "baseline"
            }
            }
        

        valid_shapes = list(shape_configs.keys())

        # If a string is given, use the same shape for all anomalies.
        if isinstance(anomaly_shapes, str):
            if anomaly_shapes not in valid_shapes:
                raise ValueError(f"Unknown anomaly shape: {anomaly_shapes}. Valid shapes are: {valid_shapes}")

            anomaly_shapes = [anomaly_shapes] * num_anomalies

        elif isinstance(anomaly_shapes, list):
            if len(anomaly_shapes) == 0:
                raise ValueError("anomaly_shapes list cannot be empty.")

            for shape in anomaly_shapes:
                if shape not in valid_shapes:
                    raise ValueError(f"Unknown anomaly shape: {shape}. Valid shapes are: {valid_shapes}")

            # Case 1: one shape in a list -> repeat it for all anomalies
            if len(anomaly_shapes) == 1:
                anomaly_shapes = anomaly_shapes * num_anomalies

            # Case 2: one shape per anomaly -> use directly
            elif len(anomaly_shapes) == num_anomalies:
                anomaly_shapes = anomaly_shapes

            # Case 3: mismatch -> raise error
            else:
                raise ValueError(
                    f"When anomaly_shapes is a list, it must either contain exactly 1 shape "
                    f"or match num_anomalies. Got {len(anomaly_shapes)} shapes for "
                    f"{num_anomalies} anomalies."
                )

        else:
            raise TypeError("anomaly_shapes must be either a string or a list of strings.")

        edge_margin_points = int(edge_margin * n)
        min_distance_points = int(min_distance * n)

        if num_anomalies > 1:
            location_used = "none"
        else:
            location_used = location

        def get_shape_profile(length, shape):
            if length <= 1:
                return np.ones(length)

            if shape == "rectangular":
                return np.ones(length)

            x = np.linspace(0, 1, length)

            if shape == "gaussian":
                center = 0.5
                width = 0.28
                profile = np.exp(-0.5 * ((x - center) / width) ** 2)
                profile = profile - profile.min()
                profile = profile / np.max(profile)

            elif shape == "triangular":
                profile = 1 - np.abs(2 * x - 1)
                profile = profile ** 1.5

            elif shape == "ramp":
                profile = x

            elif shape == "decay":
                profile = np.linspace(1, 0, length)

            else:
                raise ValueError(f"Unknown anomaly shape: {shape}")

            return profile

        def get_start_bounds(location_used, length):
            if location_used == "beginning":
                start_low = int(0.10 * n)
                start_high = int(0.30 * n)

            elif location_used == "middle":
                start_low = int(0.40 * n)
                start_high = int(0.60 * n)

            elif location_used == "end":
                start_low = int(0.70 * n)
                start_high = int(0.90 * n)

            else:
                start_low = int(0.10 * n)
                start_high = int(0.85 * n)

            latest_possible_start = n - edge_margin_points - length

            start_low = max(start_low, edge_margin_points)
            start_high = min(start_high, latest_possible_start)

            return start_low, start_high

        def interval_is_valid(start, end, selected_intervals):
            for existing_start, existing_end in selected_intervals:
                too_close_or_overlapping = not (
                    end + min_distance_points <= existing_start
                    or start >= existing_end + min_distance_points
                )

                if too_close_or_overlapping:
                    return False

            return True

        selected_intervals = []
        records = []

        for shape in anomaly_shapes:
            config = shape_configs[shape]

            min_len = max(3, int(config["length_range"][0] * n))
            max_len = max(min_len + 1, int(config["length_range"][1] * n))

            found_interval = False

            for _ in range(max_attempts):
                length = np.random.randint(min_len, max_len + 1)

                start_low, start_high = get_start_bounds(location_used, length)

                if start_high <= start_low:
                    continue

                start = np.random.randint(start_low, start_high + 1)
                end = start + length

                # First check overlap / distance condition
                if not interval_is_valid(start, end, selected_intervals):
                    continue

                # Reject visually awkward boundaries.
                # This prevents the anomaly from starting or ending exactly at an extreme jump/spike.
                boundary_window = max(5, int(0.05 * n))
                boundary_threshold = 2.5

                left = max(0, start - boundary_window)
                right = min(n, end + boundary_window)

                local_region = original_series.iloc[left:right].to_numpy()
                local_std = np.std(local_region)

                if local_std < 1e-8:
                    local_std = np.std(original_series.to_numpy())

                if local_std < 1e-8:
                    local_std = 1.0

                start_jump = abs(original_series.iloc[start] - original_series.iloc[start - 1]) if start > 0 else 0
                end_jump = abs(original_series.iloc[end] - original_series.iloc[end - 1]) if end < n else 0

                if start_jump > boundary_threshold * local_std:
                    continue

                if end_jump > boundary_threshold * local_std:
                    continue

                selected_intervals.append((start, end))
                found_interval = True
                break

            if not found_interval:
                raise ValueError(
                    f"Could not place anomaly with shape '{shape}'. "
                    f"Try reducing num_anomalies, min_distance, or anomaly length ranges."
                )

            profile = get_shape_profile(length, shape)

            local_start = max(0, start - int(0.10 * n))
            local_segment = original_series.iloc[local_start:start].to_numpy()

            if len(local_segment) > 3 and np.std(local_segment) > 1e-8:
                local_std = np.std(local_segment)
            else:
                local_std = np.std(original_series.to_numpy())

            if local_std < 1e-8:
                local_std = 1.0

            magnitude_strength = np.random.uniform(*config["magnitude_range"])
            magnitude = magnitude_strength * local_std * scale_factor
            sign = np.random.choice([-1, 1])

            anomaly_pattern = sign * magnitude * profile

            if config["method"] == "add":
                segment = series.iloc[start:end].to_numpy()
                series.iloc[start:end] = segment + anomaly_pattern

            elif config["method"] == "baseline":
                segment = series.iloc[start:end].to_numpy()

                baseline_window = max(5, int(0.03 * n))

                before_segment = series.iloc[
                    max(0, start - baseline_window):start
                ]

                after_segment = series.iloc[
                    end:min(n, end + baseline_window)
                ]

                if len(before_segment) > 0:
                    baseline_start = np.median(before_segment)
                else:
                    baseline_start = series.iloc[start]

                if len(after_segment) > 0:
                    baseline_end = np.median(after_segment)
                else:
                    baseline_end = series.iloc[end - 1]

                baseline = np.linspace(
                    baseline_start,
                    baseline_end,
                    length
                )

                segment_trend = np.linspace(
                    segment[0],
                    segment[-1],
                    length
                )

                residual = segment - segment_trend
                residual_weight = config["residual_weight"]

                series.iloc[start:end] = (
                    baseline
                    + residual_weight * residual
                    + anomaly_pattern
                )

            # Record the anomaly after it has been created
            records.append({
                "start": start,
                "end": end,
                "shape": shape,
                "magnitude": sign * magnitude,
                "magnitude_strength": magnitude_strength,
                "length": length
            })

        # This part must be outside the anomaly_shapes loop
        records = sorted(
            records,
            key=lambda item: item["start"]
        )

        selected_starts = np.array(
            [item["start"] for item in records],
            dtype=int
        )

        ends = np.array(
            [item["end"] for item in records],
            dtype=int
        )

        shapes_used = [
            item["shape"]
            for item in records
        ]

        magnitudes = [
            item["magnitude"]
            for item in records
        ]

        lengths = [
            item["length"]
            for item in records
        ]

        magnitude_strengths = [
            item["magnitude_strength"]
            for item in records
        ]

        info = {
            "type": "anomaly",
            "subtype": "collective",
            "anomaly_shapes": shapes_used,
            "num_anomalies": len(records),
            "location": location_used,
            "starts": selected_starts,
            "ends": ends,
            "lengths": lengths,
            "magnitudes": magnitudes,
            "magnitude_strengths": magnitude_strengths
        }

        df.loc[:, "data"] = series
        df.loc[:, "stationary"] = 0
        df.loc[:, "collect_anom"] = 1

        if is_loc is True:
            collect_anom_label = np.zeros(
                n,
                dtype=int
            )

            for record in records:
                start = int(record["start"])
                end = int(record["end"])

                collect_anom_label[start:end] = 1

            df.loc[:, "collect_anom_label"] = (
                collect_anom_label
            )

        return df, info

    def _get_fourier_context(
        self,
        seasonal_info,
        n
    ):
        """
        Reconstruct one existing Fourier seasonal context.

        Supports:
            - single_seasonality
            - multiple_seasonality
            - SARMA
            - SARIMA

        For SARIMA, the Fourier term belongs to the
        seasonally differenced domain, so it is
        seasonally integrated to obtain its contribution
        in the observed-series domain.
        """

        periods = seasonal_info.get("periods")

        if periods is None or len(periods) == 0:
            raise ValueError(
                "seasonal_info must contain at least "
                "one seasonal period."
            )

        # --------------------------------------------------
        # Choose one existing seasonal component
        # --------------------------------------------------
        period = random.choice(periods)

        subtype = seasonal_info.get(
            "subtype",
            ""
        ).lower()

        # --------------------------------------------------
        # Retrieve coefficients
        # --------------------------------------------------
        if subtype == "multiple_seasonality":

            all_coefficients = seasonal_info.get(
                "coefficients"
            )

            if all_coefficients is None:
                raise ValueError(
                    "No Fourier coefficients found "
                    "for multiple seasonality."
                )

            period_data = next(
                (
                    item
                    for item in all_coefficients
                    if item["period"] == period
                ),
                None
            )

            if period_data is None:
                raise ValueError(
                    f"No Fourier coefficients found "
                    f"for period={period}."
                )

            coefficients = period_data[
                "coefficients"
            ]

        elif subtype == "sarma":

            coefficients = seasonal_info.get(
                "fourier_coefficients"
            )

            if coefficients is None:
                raise ValueError(
                    "No Fourier coefficients found "
                    "for SARMA."
                )

        else:
            # single seasonality + SARIMA
            coefficients = seasonal_info.get(
                "coefficients"
            )

            if coefficients is None:
                raise ValueError(
                    f"No Fourier coefficients found "
                    f"for subtype={subtype}."
                )

        # --------------------------------------------------
        # Reconstruct Fourier term F_t
        # --------------------------------------------------
        t = np.arange(n)

        fourier_term = np.zeros(
            n,
            dtype=float
        )

        for coefficient in coefficients:

            k = coefficient["harmonic"]
            sin_coef = coefficient["sin_coef"]
            cos_coef = coefficient["cos_coef"]

            fourier_term += (
                sin_coef
                * np.sin(
                    2 * np.pi * k * t / period
                )
                +
                cos_coef
                * np.cos(
                    2 * np.pi * k * t / period
                )
            )

        scale_factor = seasonal_info.get(
            "scale_factor",
            1.0
        )

        fourier_term *= scale_factor

        # --------------------------------------------------
        # SARIMA special case
        #
        # (1 - B^s) Y_t = F_t + epsilon_t
        #
        # F_t must therefore be seasonally integrated
        # before it represents seasonal context in
        # the observed Y_t domain.
        # --------------------------------------------------
        if subtype == "sarima":

            seasonal_context = np.zeros(
                n,
                dtype=float
            )

            for i in range(n):

                if i < period:
                    seasonal_context[i] = (
                        fourier_term[i]
                    )

                else:
                    seasonal_context[i] = (
                        seasonal_context[
                            i - period
                        ]
                        + fourier_term[i]
                    )

        else:
            # Single, multiple, SARMA:
            # Fourier component is already in level domain.
            seasonal_context = fourier_term

        return period, seasonal_context
    
    def generate_contextual_anomalies(
        self,
        df,
        seasonal_info,
        num_anomalies=1,
        location=None,
        anomaly_strength=1,
        max_attempts=10,
        is_loc=None
    ):
        series_original = df["data"].copy()
        n = len(series_original)

        # --------------------------------------------------
        # Reconstruct an existing seasonal context
        # --------------------------------------------------
        period, seasonal_context = (
            self._get_fourier_context(
                seasonal_info=seasonal_info,
                n=n
            )
        )

        for attempt in range(max_attempts):
            min_distance = max(
                1,
                int((0.05 - attempt * 0.003) * n)
            )

            series = series_original.copy()

            # These are the selected peak/valley center points.
            selected_starts = []

            # These store the actual anomaly intervals.
            anomaly_intervals = []

            # --------------------------------------------------
            # Find contextual points from the actual
            # Fourier seasonal component
            # --------------------------------------------------
            peaks = np.where(
                (
                    seasonal_context[1:-1]
                    > seasonal_context[:-2]
                )
                &
                (
                    seasonal_context[1:-1]
                    > seasonal_context[2:]
                )
            )[0] + 1

            valleys = np.where(
                (
                    seasonal_context[1:-1]
                    < seasonal_context[:-2]
                )
                &
                (
                    seasonal_context[1:-1]
                    < seasonal_context[2:]
                )
            )[0] + 1

            candidate_indices = np.concatenate(
                [peaks, valleys]
            )

            # --------------------------------------------------
            # Determine candidate regions
            # --------------------------------------------------
            if num_anomalies == 1:
                if location == "beginning":
                    candidate_range = np.arange(
                        int(0.1 * n),
                        int(0.3 * n)
                    )

                elif location == "middle":
                    candidate_range = np.arange(
                        int(0.4 * n),
                        int(0.6 * n)
                    )

                elif location == "end":
                    candidate_range = np.arange(
                        int(0.7 * n),
                        int(0.9 * n)
                    )

                else:
                    candidate_range = np.arange(
                        int(0.1 * n),
                        int(0.85 * n)
                    )

                    location = "none"

            else:
                candidate_range = np.arange(
                    int(0.1 * n),
                    int(0.85 * n)
                )

                location = "none"

            candidate_indices = np.array([
                i
                for i in candidate_indices
                if i in candidate_range
            ])

            if len(candidate_indices) == 0:
                print(
                    f"[Attempt {attempt + 1}] "
                    f"No candidates found for "
                    f"n={n}, period={period}"
                )
                continue

            # --------------------------------------------------
            # Select anomaly centers with spacing
            # --------------------------------------------------
            candidates = candidate_indices.copy()
            np.random.shuffle(candidates)

            for center in candidates:
                if all(
                    abs(center - previous_center)
                    >= min_distance
                    for previous_center
                    in selected_starts
                ):
                    selected_starts.append(center)

                if (
                    len(selected_starts)
                    == num_anomalies
                ):
                    break

            # Fill remaining anomalies without spacing
            # if necessary
            if len(selected_starts) < num_anomalies:
                remaining = list(
                    set(candidate_indices)
                    - set(selected_starts)
                )

                np.random.shuffle(remaining)

                for center in remaining:
                    selected_starts.append(center)

                    if (
                        len(selected_starts)
                        == num_anomalies
                    ):
                        break

            if len(selected_starts) == 0:
                continue

            # --------------------------------------------------
            # Apply contextual anomalies
            # --------------------------------------------------
            for center in selected_starts:
                anomaly_length = min(
                    max(
                        int(period * 0.5),
                        10
                    ),
                    int(0.2 * n)
                )

                start = max(
                    0,
                    center - anomaly_length // 2
                )

                end = min(
                    n,
                    start + anomaly_length
                )

                anomaly_intervals.append(
                    (start, end)
                )

                # Actual Fourier seasonal context
                # associated with the chosen period.
                local_season = (
                    seasonal_context[start:end]
                )

                # Locally violate/invert the expected
                # seasonal context.
                series.iloc[start:end] -= (
                    2
                    * local_season
                    * anomaly_strength
                )

            # Successful generation
            break

        else:
            print(
                f"generate_contextual_anomalies "
                f"failed for n={n}"
            )

            return df, None

        # --------------------------------------------------
        # Sort anomaly intervals
        # --------------------------------------------------
        anomaly_intervals = sorted(
            anomaly_intervals,
            key=lambda interval: interval[0]
        )

        anomaly_starts = np.array([
            start
            for start, end in anomaly_intervals
        ])

        anomaly_ends = np.array([
            end
            for start, end in anomaly_intervals
        ])

        # --------------------------------------------------
        # Period meaning
        # --------------------------------------------------
        period_meanings = seasonal_info.get(
            "period_meanings",
            {}
        )

        period_meaning = period_meanings.get(
            period,
            self.get_period_meanings(period)
        )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------
        info = {
            "type": "anomaly",
            "subtype": "contextual",
            "num_anomalies": len(
                anomaly_intervals
            ),
            "location": location,
            "starts": anomaly_starts,
            "ends": anomaly_ends,

            # The seasonal component whose context
            # was violated.
            "periods": [period],
            "period_meanings": {
                period: period_meaning
            }
        }

        df.loc[:, "data"] = series
        df.loc[:, "stationary"] = 0
        df.loc[:, "context_anom"] = 1
        df.loc[:, "seasonal"] = 1

        # --------------------------------------------------
        # Create location labels only when requested
        # --------------------------------------------------
        if is_loc is True:
            context_anom_label = np.zeros(
                n,
                dtype=int
            )

            for start, end in anomaly_intervals:
                context_anom_label[start:end] = 1

            df.loc[
                :,
                "context_anom_label"
            ] = context_anom_label

        return df, info
    
    #TRENDS - DETERMINISTIC TRENDS

    def generate_deterministic_trend_linear(self, df, sign = None, slope= None, noise_std = None, intercept = 1, scale_factor = 1):
        series = df['data'].copy()
        sign = sign if sign is not None else np.random.choice([-1,1])
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        slope = slope if slope is not None else sign * random.uniform(0.05, 0.5) / (len(series) / 100)
        trend = intercept + slope * np.arange(len(series)) + np.random.normal(0, noise_std, len(series))
        series += trend * scale_factor
        info = {'type' : 'trend', 'subtype': 'deterministic_linear', 'sign': sign, 'slope': slope, 'intercept': intercept}
        df.loc[:,'data'] = series
        df.loc[:,'stationary'] = 0
        if sign > 0:
            df.loc[:,'det_lin_up'] = 1
        else:
            df.loc[:,'det_lin_down'] = 1
        return df, info

    def generate_deterministic_trend_quadratic(self, df, sign=None, a=None, b=None, c=None,noise_std=None, scale_factor=1,asymmetric=False, location="center"):
        series = df['data'].copy()
        sign = sign if sign in [-1, 1] else random.choice([-1, 1])
        length = len(series)
        t = np.linspace(-1, 1, length)
    
        # Choose strength of curvature
        a = a if a is not None else sign * random.uniform(2.0, 5.0)
    
        # Compute linear term to move vertex
        if location == "center":
            b = 0
        elif location == "left":
            b = -2 * a * (-0.5)  # vertex at t = -0.5
        elif location == "right":
            b = -2 * a * (0.5)   # vertex at t = +0.5
        else:
            raise ValueError("location must be 'center', 'left', or 'right'")
    
        c = c if c is not None else 0
    
        trend = (a * t**2 + b * t + c) * scale_factor
    
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_std, length)

        info = {'type' : 'trend', 'subtype': 'deterministic_quadratic','sign': sign, 'a': a, 'b': b, 'c': c}
    
        series += trend + noise
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'det_quad'] = 1
        return df, info

    def generate_deterministic_trend_cubic(self, df, sign=None, amplitude=10, noise_std=None,scale_factor=1, asymmetric=False, location="center"):
        series = df['data'].copy()
        sign = sign if sign in [-1, 1] else random.choice([-1, 1])
        length = len(series)
        t = np.linspace(-1, 1, length)
    
        a = 1.0  # fixed cubic term
        c = -1.0  # linear slope for S shape
    
        # Inflection point: t_i = -b / (3a) → solve for b
        if location == "center":
            b = 0
        elif location == "left":
            b = -3 * a * (-0.5)  # inflection at t = -0.5
        elif location == "right":
            b = -3 * a * (0.5)   # inflection at t = +0.5
        else:
            raise ValueError("location must be 'center', 'left', or 'right'")
    
        # If asymmetric override is also set, add to b
        if asymmetric:
            b += sign * random.uniform(0.5, 2.0)
    
        # Final trend
        trend = (a * t**3 + b * t**2 + c * t) * amplitude
    
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, length)
    
        series += trend * scale_factor + noise

        info = {'type' : 'trend', 'subtype': 'deterministic_cubic','sign': sign, 'a': a, 'b': b}
        
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'det_cubic'] = 1
        return df, info

    def generate_deterministic_trend_exponential(self, df, sign=None, a=None, b=None, noise_std=None, scale_factor=1):
        series = df['data'].copy()
        sign = sign if sign in [-1, 1] else random.choice([-1, 1])
        length = len(series)
        a = a if a is not None else random.uniform(1.0, 2.0)
        b = b if b is not None else random.uniform(1.5, 3.0)
        t = np.linspace(0, 2, len(series))

        if sign == 1:
            noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 0.5)
            trend = a * np.exp(b * t)
            scale_factor = 1
        else:
            noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
            trend = a * np.exp(-b * t)
            scale_factor = 5
            
        trend *= scale_factor
        noise = np.random.normal(0, noise_std, length)
    
        series += trend + noise*3

        info = {'type' : 'trend', 'subtype': 'deterministic_exponential','sign': sign, 'a': a, 'b': b}
    
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'det_exp'] = 1
        return df, info

    def generate_deterministic_trend_damped(self, df, sign=None, a=None, b=None, damping_rate=None, noise_std=None, scale_factor=1):
        series = df['data'].copy()
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        sign = sign if sign is not None else random.choice([-1, 1])
        a = a if a is not None else sign * np.random.normal(loc=1.0, scale=0.2)
        b = b if b is not None else np.random.normal(loc=0.1, scale=0.05)
        damping_rate = damping_rate if damping_rate is not None else random.uniform(0.01, 0.005)
        t = np.arange(len(series))
        noise = np.random.normal(0, noise_std, len(series))
        trend = (a * t + b) * np.exp(-damping_rate * t) * scale_factor + noise
        series += trend
        info = {'type' : 'trend', 'subtype': 'deterministic_damped','damping_rate': damping_rate, 'a': a, 'b': b}
        df.loc[:, 'data'] = series
        df.loc[:,'stationary'] = 0
        df.loc[:, 'det_damped'] = 1
        return df,info

    #TRENDS - STOCHASTIC TRENDS

    def generate_stochastic_trend(self, kind='rw', d = 1, const=False, drift=None, noise_std=1.0):
        t = np.arange(self.length)
        noise = np.random.normal(0, noise_std, self.length)
    
        if kind == 'rw':
            info = {'type': 'trend', 'subtytpe': 'random_walk'}
            series = np.cumsum(noise)

        elif kind == 'rwd':
            if drift is None:
                drift = np.random.uniform(0.01, 0.1)
            info = {'type': 'trend', 'subtype': 'random_walk_with_drift', 'drift': drift}
            series = drift * t + np.cumsum(noise)
    
        elif kind == 'ari':
            series, info = self.generate_ari_series(length=self.length, d = d, const = const)
    
        elif kind == 'ima':
            series, info = self.generate_ima_series(length=self.length, d = d, const = const)
    
        elif kind == 'arima':
            series, info = self.generate_arima_series(length=self.length, d = d, const = const)
    
        else:
            raise ValueError("Invalid kind. Choose from 'rw', 'rwd', 'ari', 'ima', or 'arima'.")

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': (np.zeros(self.length)).astype(int),
            'seasonal': np.zeros(self.length).astype(int),
        })
        return df, info

    # PERIOD HELPERS

    def get_calendar_periods(self):
        """
        Calendar-meaningful seasonal periods for different sampling frequencies.
        Period always means: number of observations per cycle.
        """
        return {
            "monthly": {
                3: "quarterly cycle",
                6: "semiannual cycle",
                12: "annual cycle"
            },
            "quarterly": {
                4: "annual cycle"
            },
            "daily": {
                7: "weekly cycle",
                30: "monthly-ish cycle",
                90: "quarterly-ish cycle",
                180: "semiannual-ish cycle",
                365: "annual cycle"
            },
            "weekly": {
                4: "monthly-ish cycle",
                13: "quarterly cycle",
                26: "semiannual cycle",
                52: "annual cycle"
            },
            "business_daily": {
                5: "weekly cycle",
                21: "monthly-ish cycle",
                63: "quarterly-ish cycle",
                126: "semiannual-ish cycle",
                252: "annual-ish cycle"
            },
            "hourly": {
                24: "daily cycle",
                168: "weekly cycle"
            }
        }

    def get_all_calendar_periods(self):
        calendar_periods = self.get_calendar_periods()

        all_periods = sorted(
            set(
                period
                for sampling_dict in calendar_periods.values()
                for period in sampling_dict.keys()
            )
        )

        return all_periods

    def get_period_meanings(self, period):
        """
        Returns all possible calendar interpretations of a period.
        Example:
            period=4 can mean:
            - quarterly data: annual cycle
            - weekly data: monthly-ish cycle
        """
        calendar_periods = self.get_calendar_periods()
        meanings = []

        for sampling_frequency, period_dict in calendar_periods.items():
            if period in period_dict:
                meanings.append({
                    "sampling_frequency": sampling_frequency,
                    "meaning": period_dict[period]
                })

        return meanings

    def get_valid_calendar_periods(
        self,
        allowed_periods=None,
        min_cycles=6
    ):
        """
        Filters periods according to series length.

        min_cycles=6 means:
            selected period should appear at least about 6 times in the series.
        """
        n = self.length

        if allowed_periods is None:
            allowed_periods = self.get_all_calendar_periods()

        allowed_periods = sorted(set(int(p) for p in allowed_periods))

        max_period = n // min_cycles

        valid_periods = [
            p for p in allowed_periods
            if p <= max_period
        ]

        return valid_periods

    def choose_calendar_period(
        self,
        period=None,
        allowed_periods=None,
        min_cycles=6
    ):
        """
        Chooses or validates a period using calendar-meaningful periods.
        """
        n = self.length

        if allowed_periods is None:
            allowed_periods = self.get_all_calendar_periods()

        allowed_periods = sorted(set(int(p) for p in allowed_periods))
        valid_periods = self.get_valid_calendar_periods(
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        if len(valid_periods) == 0:
            raise ValueError(
                f"No valid period found for length={n}. "
                f"Allowed periods are {allowed_periods}, but min_cycles={min_cycles} requires period <= {n // min_cycles}."
            )

        if period is None:
            period = random.choice(valid_periods)
        else:
            period = int(period)

            if period not in allowed_periods:
                raise ValueError(
                    f"period={period} is not in allowed calendar periods: {allowed_periods}"
                )

            if period not in valid_periods:
                raise ValueError(
                    f"period={period} is too large for length={n} with min_cycles={min_cycles}. "
                    f"Valid periods are {valid_periods}."
                )

        return period, valid_periods

    def normalize_period_list(self, periods):
        """
        Converts period input into a list.
        """
        if periods is None:
            return None

        if isinstance(periods, (int, np.integer)):
            return [int(periods)]

        return [int(p) for p in periods]

# SEASONALITY

    def generate_single_seasonality(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1,
        num_harmonics=1,
        allowed_periods=None,
        min_cycles=6
    ):
        n = self.length
        t = np.arange(n)

        series = np.random.normal(loc=0.0, scale=0.2, size=n)

        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)

        period, valid_periods = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        base_std = np.std(series)
        amplitude = amplitude if amplitude is not None else base_std * np.random.uniform(0.5, 2.5)

        seasonality = np.zeros(n)
        coefficients = []

        for k in range(1, num_harmonics + 1):
            A_k = amplitude * np.random.uniform(0.5, 1.0) / k
            B_k = amplitude * np.random.uniform(0.5, 1.0) / k

            seasonality += A_k * np.sin(2 * np.pi * k * t / period)
            seasonality += B_k * np.cos(2 * np.pi * k * t / period)

            coefficients.append({
                "harmonic": k,
                "sin_coef": A_k,
                "cos_coef": B_k
            })

        seasonality += np.random.normal(0, noise_std, size=n)

        series += seasonality * scale_factor

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "single_seas": np.ones(n).astype(int)
        })

        info = {
            "type": "seasonal",
            "subtype": "single_seasonality",
            "periods": [period],
            "period_meanings": {
                period: self.get_period_meanings(period)
            },
            "amplitude": amplitude,
            "noise_std": noise_std,
            "scale_factor": scale_factor,
            "num_harmonics": num_harmonics,
            "coefficients": coefficients
        }

        return df, info
    
    def generate_multiple_seasonality(
        self,
        num_components=2,
        periods=None,
        amplitudes=None,
        noise_std=None,
        scale_factor=3,
        num_harmonics=1,
        allowed_periods=None,
        min_cycles=6
    ):
        n = self.length
        t = np.arange(n)

        series = np.random.normal(loc=0.0, scale=0.2, size=n)

        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)

        if allowed_periods is None:
            allowed_periods = self.get_all_calendar_periods()

        allowed_periods = sorted(set(int(p) for p in allowed_periods))

        valid_periods = self.get_valid_calendar_periods(
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        if len(valid_periods) < 2 and periods is None:
            raise ValueError(
                f"Multiple seasonality needs at least 2 valid periods. "
                f"For length={n}, valid periods are {valid_periods}."
            )

        periods = self.normalize_period_list(periods)

        if periods is None:
            periods = random.sample(
                valid_periods,
                min(num_components, len(valid_periods))
            )
        else:
            for p in periods:
                if p not in allowed_periods:
                    raise ValueError(
                        f"period={p} is not in allowed calendar periods: {allowed_periods}"
                    )

                if p not in valid_periods:
                    raise ValueError(
                        f"period={p} is too large for length={n} with min_cycles={min_cycles}. "
                        f"Valid periods are {valid_periods}."
                    )

            if len(periods) < 2:
                raise ValueError(
                    "Multiple seasonality should have at least 2 periods. "
                    "Pass something like periods=[7, 30] or periods=[12, 24]."
                )

        if amplitudes is None:
            base_std = np.std(series)
            amplitudes = [
                base_std * np.random.uniform(0.5, 2.0)
                for _ in periods
            ]
        else:
            if len(amplitudes) != len(periods):
                raise ValueError(
                    f"Length of amplitudes must match length of periods. "
                    f"Got {len(amplitudes)} amplitudes and {len(periods)} periods."
                )

        seasonality = np.zeros(n)

        periods_meta = []
        amplitudes_meta = []
        coefficients_meta = []

        for period, amplitude in zip(periods, amplitudes):
            period_coefficients = []

            for k in range(1, num_harmonics + 1):
                A_k = amplitude * np.random.uniform(0.5, 1.0) / k
                B_k = amplitude * np.random.uniform(0.5, 1.0) / k

                seasonality += A_k * np.sin(2 * np.pi * k * t / period)
                seasonality += B_k * np.cos(2 * np.pi * k * t / period)

                period_coefficients.append({
                    "harmonic": k,
                    "sin_coef": A_k,
                    "cos_coef": B_k
                })

            periods_meta.append(period)
            amplitudes_meta.append(amplitude)
            coefficients_meta.append({
                "period": period,
                "coefficients": period_coefficients
            })

        seasonality += np.random.normal(0, noise_std, size=n)

        series += seasonality * scale_factor

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "multiple_seas": np.ones(n).astype(int)
        })

        info = {
            "type": "seasonal",
            "subtype": "multiple_seasonality",
            "periods": periods_meta,
            "period_meanings": {
                p: self.get_period_meanings(p)
                for p in periods_meta
            },
            "amplitudes": amplitudes_meta,
            "noise_std": noise_std,
            "scale_factor": scale_factor,
            "num_harmonics": num_harmonics,
            "coefficients": coefficients_meta
        }

        return df, info

    def generate_sarima_series(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1,
        initial_std=0.2,
        num_harmonics=1,
        allowed_periods=None,
        min_cycles=6
    ):
        """
        Seasonal unit root case with deterministic Fourier seasonal difference.

        Model:
            Y_t - Y_{t-s} = Fourier(t) + noise
        """

        n = self.length
        t = np.arange(n)

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        if n <= period:
            raise ValueError("Series length must be larger than the seasonal period.")

        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 0.20)

        if amplitude is None:
            base_series = np.random.normal(loc=0.0, scale=0.2, size=n)
            amplitude = np.std(base_series) * np.random.uniform(0.5, 2.5)

        seasonal_difference = np.zeros(n)
        coefficients = []

        for k in range(1, num_harmonics + 1):
            A_k = amplitude * np.random.uniform(0.5, 1.0) / k
            B_k = amplitude * np.random.uniform(0.5, 1.0) / k

            seasonal_difference += A_k * np.sin(2 * np.pi * k * t / period)
            seasonal_difference += B_k * np.cos(2 * np.pi * k * t / period)

            coefficients.append({
                "harmonic": k,
                "sin_coef": A_k,
                "cos_coef": B_k
            })

        seasonal_difference += np.random.normal(0, noise_std, size=n)
        seasonal_difference = seasonal_difference * scale_factor

        series = np.zeros(n)
        series[:period] = np.random.normal(0, initial_std, size=period)

        for i in range(period, n):
            series[i] = series[i - period] + seasonal_difference[i]

        actual_seasonal_difference = np.full(n, np.nan)
        actual_seasonal_difference[period:] = series[period:] - series[:-period]

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "seasonal_diff": actual_seasonal_difference,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "sarima": np.ones(n).astype(int)
        })

        info = {
            "type": "seasonal",
            "subtype": "SARIMA",
            "periods": [period],
            "period_meanings": {
                period: self.get_period_meanings(period)
            },
            "diff": 0,
            "seasonal_diff": 1,
            "unit_root": "seasonal_unit_root",
            "amplitude": amplitude,
            "noise_std": noise_std,
            "scale_factor": scale_factor,
            "initial_std": initial_std,
            "num_harmonics": num_harmonics,
            "coefficients": coefficients
        }

        return df, info

    def generate_sarma_series(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1,
        error_scale=None,
        num_harmonics=1,
        order_range=(1, 2),
        seasonal_order_range=(1, 1),
        coef_range=(-0.4, 0.4),
        seasonal_coef_range=(-0.4, 0.4),
        allowed_periods=None,
        min_cycles=6,
        max_attempts=1000
    ):
        """
        Generates a SARMA-like seasonal series with deterministic seasonality.

        Model:
            Y_t = f_t + u_t + e_t

        where:
            f_t = deterministic Fourier seasonality
            u_t = stationary SARMA error process
            e_t = small observation noise

        There is no unit root and no differencing.
        """

        n = self.length
        t = np.arange(n)

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        s = period

        if amplitude is None:
            base_series = np.random.normal(loc=0.0, scale=0.2, size=n)
            amplitude = np.std(base_series) * np.random.uniform(0.8, 2.5)

        deterministic_seasonality = np.zeros(n)
        fourier_coefficients = []

        for k in range(1, num_harmonics + 1):
            A_k = amplitude * np.random.uniform(0.5, 1.0) / k
            B_k = amplitude * np.random.uniform(0.5, 1.0) / k

            deterministic_seasonality += A_k * np.sin(2 * np.pi * k * t / s)
            deterministic_seasonality += B_k * np.cos(2 * np.pi * k * t / s)

            fourier_coefficients.append({
                "harmonic": k,
                "sin_coef": A_k,
                "cos_coef": B_k
            })

        deterministic_seasonality = deterministic_seasonality * scale_factor

        sarma_error = None

        for _ in range(max_attempts):
            ar_order = np.random.randint(order_range[0], order_range[1] + 1)
            ma_order = np.random.randint(order_range[0], order_range[1] + 1)

            seasonal_ar_order = np.random.randint(
                seasonal_order_range[0],
                seasonal_order_range[1] + 1
            )
            seasonal_ma_order = np.random.randint(
                seasonal_order_range[0],
                seasonal_order_range[1] + 1
            )

            ar_coefs = np.random.uniform(coef_range[0], coef_range[1], ar_order)
            ma_coefs = np.random.uniform(coef_range[0], coef_range[1], ma_order)

            seasonal_ar_coefs = np.random.uniform(
                seasonal_coef_range[0],
                seasonal_coef_range[1],
                seasonal_ar_order
            )

            seasonal_ma_coefs = np.random.uniform(
                seasonal_coef_range[0],
                seasonal_coef_range[1],
                seasonal_ma_order
            )

            nonseasonal_ar = np.r_[1, -ar_coefs]
            nonseasonal_ma = np.r_[1, ma_coefs]

            seasonal_ar = np.zeros(seasonal_ar_order * s + 1)
            seasonal_ar[0] = 1
            for i, coef in enumerate(seasonal_ar_coefs, start=1):
                seasonal_ar[i * s] = -coef

            seasonal_ma = np.zeros(seasonal_ma_order * s + 1)
            seasonal_ma[0] = 1
            for i, coef in enumerate(seasonal_ma_coefs, start=1):
                seasonal_ma[i * s] = coef

            ar_poly = np.convolve(nonseasonal_ar, seasonal_ar)
            ma_poly = np.convolve(nonseasonal_ma, seasonal_ma)

            arma_process = ArmaProcess(ar_poly, ma_poly)

            if arma_process.isstationary and arma_process.isinvertible:
                burnin = max(200, 8 * s)

                sarma_error = arma_process.generate_sample(
                    nsample=n,
                    burnin=burnin,
                    scale=1.0
                )

                error_std = np.std(sarma_error)
                if error_std > 1e-8:
                    sarma_error = (sarma_error - np.mean(sarma_error)) / error_std

                break

        if sarma_error is None:
            raise RuntimeError("Could not generate valid stationary SARMA error.")

        if error_scale is None:
            error_scale = amplitude * np.random.uniform(0.25, 0.60)

        sarma_error = sarma_error * error_scale

        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        observation_noise = np.random.normal(0, noise_std, size=n)

        series = deterministic_seasonality + sarma_error + observation_noise

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "sarma": np.ones(n).astype(int)
        })

        info = {
            "type": "seasonal",
            "subtype": "SARMA",
            "periods": [s],
            "period_meanings": {
                s: self.get_period_meanings(s)
            },
            "diff": 0,
            "seasonal_diff": 0,
            "unit_root": "none",
            "amplitude": amplitude,
            "noise_std": noise_std,
            "scale_factor": scale_factor,
            "num_harmonics": num_harmonics,
            "fourier_coefficients": fourier_coefficients,
            "ar_order": ar_order,
            "ma_order": ma_order,
            "seasonal_ar_order": seasonal_ar_order,
            "seasonal_ma_order": seasonal_ma_order,
            "ar_coefs": ar_coefs,
            "ma_coefs": ma_coefs,
            "seasonal_ar_coefs": seasonal_ar_coefs,
            "seasonal_ma_coefs": seasonal_ma_coefs
        }

        return df, info

    #VOLATILITY

    def generate_volatility(self, kind = None):
        if kind == 'arch':
            series, info = self.generate_arch_series(self.length)
        elif kind == 'garch':
            series, info = self.generate_garch_series(self.length)
        elif kind == 'egarch':
            series, info = self.generate_egarch_series(self.length)
        elif kind == 'aparch':
            series, info = self.generate_aparch_series(self.length)

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': (np.zeros(self.length)).astype(int),
            'seasonal': np.zeros(self.length).astype(int),
        })
        return df, info


    def generate_seasonality_from_base_series(
        self,
        kind=None,
        num_components=2,
        period=None
    ):
        """
        Generates seasonal base series from scratch.

        kind:
            "single"   -> deterministic single Fourier seasonality
            "multiple" -> deterministic multiple Fourier seasonality
            "sarima"   -> seasonal unit root:
                          Y_t - Y_{t-s} = Fourier(t) + noise
            "sarma"    -> deterministic Fourier seasonality with SARMA errors
        """

        if kind is None:
            kind = random.choice(["single", "multiple", "sarma", "sarima"])

        if kind == "single":
            df, info = self.generate_single_seasonality(
                period=period,
                num_harmonics=1
            )

        elif kind == "multiple":
            periods = self.normalize_period_list(period)

            df, info = self.generate_multiple_seasonality(
                num_components=num_components,
                periods=periods,
                num_harmonics=1
            )

        elif kind == "sarima":
            df, info = self.generate_sarima_series(
                period=period,
                num_harmonics=1
            )

        elif kind == "sarma":
            df, info = self.generate_deterministic_sarma_series(
                period=period,
                num_harmonics=1
            )

        else:
            raise ValueError(
                "Invalid kind. Choose from 'single', 'multiple', 'sarma' or 'sarima'."
            )

        return df, info

    #STRUCTURAL BREAKS
    
    def generate_mean_shift(self, df, num_breaks=1, scale_factor=1, signs=None, location=None, 
                            noise_std=None, seasonal_period=None, slope=None, intercept=None, is_loc=None):
        series = df['data'].copy()
        n = len(series)
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        min_distance = 0.1 * n
        created_breaks = []
        magnitudes = []
        info = []
        
        if seasonal_period is None:
            seasonal_component = np.zeros(n)
            shift_target = series.copy()

        elif isinstance(seasonal_period, int):
            stl = STL(series, period=seasonal_period, robust=True)
            result = stl.fit()

            seasonal_component = result.seasonal
            shift_target = series - seasonal_component

        elif isinstance(seasonal_period, (list, tuple)):
            mstl = MSTL(series, periods=seasonal_period)
            result = mstl.fit()

            seasonal_component = result.seasonal
            shift_target = series - seasonal_component.sum(axis=1)

        else:
            raise ValueError("seasonal_period must be None, an int, or a list/tuple of ints.")
        # Decide break points
        if num_breaks == 1 and location in ["beginning", "middle", "end"]:
            if location == "beginning":
                break_points = [np.random.randint(int(0.1 * n), int(0.3 * n))]
            elif location == "middle":
                break_points = [np.random.randint(int(0.4 * n), int(0.6 * n))]
            elif location == "end":
                break_points = [np.random.randint(int(0.7 * n), int(0.9 * n))]
        else:
            candidates = np.arange(int(0.1 * n), int(0.9 * n))
            break_points = []
            while len(break_points) < num_breaks and len(candidates) > 0:
                point = np.random.choice(candidates)
                if isinstance(seasonal_period, int):
                    phase = point % seasonal_period
                    point -= phase
                elif isinstance(seasonal_period, (list, tuple)):
                    sp = np.random.choice(seasonal_period)
                    phase = point % sp
                    point -= phase
                if point not in break_points:
                    break_points.append(point)
                    candidates = candidates[np.abs(candidates - point) >= min_distance]
            break_points = sorted(break_points)

        if signs is None or len(signs) != len(break_points):
            raise ValueError("signs must be a list with the same length as the number of breaks.")

        info = {'type': 'structural_break', 'subtype': 'mean_shift', 'num_breaks':num_breaks, 'location' : location}
        
        prev_point = 0
        # Apply shifts
        for i, break_point in enumerate(break_points):
            local_std = np.std(shift_target[prev_point:break_point])
            magnitude = np.random.uniform(1.5, 3) * local_std
            magnitudes.append(magnitude)
            level_shift = signs[i] * magnitude
            shift_target[break_point:] += level_shift * scale_factor 
            created_breaks.append(break_point)
            prev_point = break_point

        info['shift_indices'] = created_breaks
        info['shift_magnitudes'] = magnitudes
    
        # Reconstruct series
        if seasonal_period is None:
            series = shift_target
        elif isinstance(seasonal_period, int):
            series = shift_target + seasonal_component
        elif isinstance(seasonal_period, (list, tuple)):
            series = shift_target + seasonal_component.sum(axis=1)

        noise = np.random.normal(0, noise_std, n)
        series += noise

        df.loc[:,'data'] = series
        df.loc[:,'stationary'] = 0

        if is_loc is True:
            mean_shift_label = np.zeros(n, dtype=int)

            for regime_number, break_point in enumerate(
                sorted(created_breaks),
                start=1
            ):
                mean_shift_label[break_point:] = regime_number

            df.loc[:, "mean_shift_label"] = mean_shift_label

        return df, info

    def generate_variance_shift(
        self,
        df,
        num_breaks=1,
        scale_factor=1,
        signs=None,
        location=None,
        seasonal_period=None,
        slope=None,
        intercept=None,
        is_loc=None
    ):
        series = df["data"].copy()
        n = len(series)

        min_distance = 0.1 * n
        created_breaks = []
        variance_change_factors = []

        if seasonal_period is None:
            seasonal_component = np.zeros(n)

            if slope is not None and intercept is not None:
                trend_component = intercept + slope * np.arange(n)
                residual_component = series - trend_component
            else:
                trend_component = np.zeros(n)
                residual_component = series.copy()

        elif isinstance(seasonal_period, int):
            stl = STL(
                series,
                period=seasonal_period,
                robust=True
            )
            result = stl.fit()

            trend_component = result.trend
            seasonal_component = result.seasonal
            residual_component = result.resid

        elif isinstance(seasonal_period, (list, tuple)):
            mstl = MSTL(
                series,
                periods=seasonal_period
            )
            result = mstl.fit()

            trend_component = result.trend
            seasonal_component = result.seasonal
            residual_component = result.resid

        else:
            raise ValueError(
                "seasonal_period must be None, "
                "an int, or a list/tuple of ints."
            )

        # Decide break points
        if (
            num_breaks == 1
            and location in ["beginning", "middle", "end"]
        ):
            if location == "beginning":
                break_points = [
                    np.random.randint(
                        int(0.1 * n),
                        int(0.3 * n)
                    )
                ]

            elif location == "middle":
                break_points = [
                    np.random.randint(
                        int(0.4 * n),
                        int(0.6 * n)
                    )
                ]

            elif location == "end":
                break_points = [
                    np.random.randint(
                        int(0.7 * n),
                        int(0.9 * n)
                    )
                ]

        else:
            candidates = np.arange(
                int(0.1 * n),
                int(0.9 * n)
            )

            break_points = []

            while (
                len(break_points) < num_breaks
                and len(candidates) > 0
            ):
                point = np.random.choice(candidates)

                if isinstance(seasonal_period, int):
                    phase = point % seasonal_period
                    point -= phase

                elif isinstance(
                    seasonal_period,
                    (list, tuple)
                ):
                    sp = np.random.choice(seasonal_period)
                    phase = point % sp
                    point -= phase

                if point not in break_points:
                    break_points.append(point)

                    candidates = candidates[
                        np.abs(candidates - point)
                        >= min_distance
                    ]

            break_points = sorted(break_points)

        if signs is None or len(signs) != len(break_points):
            raise ValueError(
                "signs must be a list with the same "
                "length as the number of breaks."
            )

        info = {
            "type": "structural_break",
            "subtype": "variance_shift",
            "num_breaks": len(break_points),
            "location": location
        }

        # Apply variance shifts
        for i, break_point in enumerate(break_points):
            variance_factor = np.random.uniform(1.5, 3)
            variance_change_factors.append(variance_factor)

            if signs[i] > 0:
                residual_component[break_point:] *= (
                    variance_factor * scale_factor
                )

            elif signs[i] < 0:
                residual_component[break_point:] /= (
                    variance_factor * scale_factor
                )

            created_breaks.append(break_point)

        # Reconstruct series
        if seasonal_period is None:
            series = (
                trend_component
                + residual_component
            )

        elif isinstance(seasonal_period, int):
            series = (
                trend_component
                + seasonal_component
                + residual_component
            )

        elif isinstance(seasonal_period, (list, tuple)):
            series = (
                trend_component
                + seasonal_component.sum(axis=1)
                + residual_component
            )

        info["shift_indices"] = created_breaks
        info["shift_magnitudes"] = variance_change_factors

        df.loc[:, "data"] = series
        df.loc[:, "stationary"] = 0

        # Create structural-break regime labels only when requested
        if is_loc is True:
            variance_shift_label = np.zeros(
                n,
                dtype=int
            )

            for regime_number, break_point in enumerate(
                sorted(created_breaks),
                start=1
            ):
                variance_shift_label[
                    break_point:
                ] = regime_number

            df.loc[
                :,
                "variance_shift_label"
            ] = variance_shift_label

        return df, info

    def generate_trend_shift(
        self,
        df,
        location="middle",
        num_breaks=1,
        scale_factor=1,
        change_types=None,
        slope=None,
        intercept=None,
        seasonal_period=None,
        noise_std=None,
        is_loc=None
    ):
        series = df["data"].copy()
        n = len(series)

        min_distance = 0.1 * n

        noise_std = (
            noise_std
            if noise_std is not None
            else np.random.uniform(0.01, 0.05)
        )

        created_breaks = []
        created_change_types = []

        if slope is None or intercept is None:
            raise ValueError(
                "slope and intercept must be provided for trend shift."
            )

        if seasonal_period is None:
            original_trend = intercept + slope * np.arange(n)
            residual_component = series - original_trend

        elif isinstance(seasonal_period, int):
            stl = STL(
                series,
                period=seasonal_period,
                robust=True
            )
            result = stl.fit()

            seasonal_component = result.seasonal
            residual_component = result.resid

        elif isinstance(seasonal_period, (list, tuple)):
            mstl = MSTL(
                series,
                periods=seasonal_period
            )
            result = mstl.fit()

            seasonal_component = result.seasonal
            residual_component = result.resid

        else:
            raise ValueError(
                "seasonal_period must be None, "
                "an int, or a list/tuple of ints."
            )

        # Decide break points
        if (
            num_breaks == 1
            and location in ["beginning", "middle", "end"]
        ):
            if location == "beginning":
                break_points = [
                    np.random.randint(
                        int(0.1 * n),
                        int(0.3 * n)
                    )
                ]

            elif location == "middle":
                break_points = [
                    np.random.randint(
                        int(0.4 * n),
                        int(0.6 * n)
                    )
                ]

            elif location == "end":
                break_points = [
                    np.random.randint(
                        int(0.7 * n),
                        int(0.9 * n)
                    )
                ]

        else:
            candidates = np.arange(
                int(0.1 * n),
                int(0.9 * n)
            )

            break_points = []

            while (
                len(break_points) < num_breaks
                and len(candidates) > 0
            ):
                point = np.random.choice(candidates)

                if isinstance(seasonal_period, int):
                    phase = point % seasonal_period
                    point -= phase

                elif isinstance(seasonal_period, (list, tuple)):
                    sp = np.random.choice(seasonal_period)
                    phase = point % sp
                    point -= phase

                if point not in break_points:
                    break_points.append(point)

                    candidates = candidates[
                        np.abs(candidates - point)
                        >= min_distance
                    ]

            break_points = sorted(break_points)

        # Validate change_types input
        if (
            change_types is None
            or len(change_types) != len(break_points)
        ):
            raise ValueError(
                "change_types must be a list with the same "
                "length as the number of breaks."
            )

        # Initialize trend array
        current_slope = slope
        current_level = intercept

        trend = np.zeros(n)
        prev_point = 0

        info = {
            "type": "structural_break",
            "subtype": "trend_shift",
            "num_breaks": len(break_points),
            "location": location
        }

        # Construct piecewise trend
        for i, break_point in enumerate(break_points + [n]):
            slope_change_factor = np.random.uniform(1.5, 4.5)
            segment_length = break_point - prev_point

            if segment_length > 0:
                segment_trend = (
                    current_level
                    + current_slope * np.arange(segment_length)
                )

                trend[prev_point:break_point] = segment_trend
                current_level = segment_trend[-1]

            if break_point == n:
                break

            change_type = change_types[i]

            if change_type == "direction_change":
                current_slope = -current_slope

            elif change_type == "magnitude_change":
                current_slope = (
                    current_slope
                    * slope_change_factor
                    * scale_factor
                )

            elif change_type == "direction_and_magnitude_change":
                current_slope = (
                    -current_slope
                    * slope_change_factor
                    * scale_factor
                )

            else:
                raise ValueError(
                    "Invalid change_type: "
                    + str(change_type)
                )

            created_breaks.append(break_point)
            created_change_types.append(change_type)
            prev_point = break_point

        info["shift_indices"] = created_breaks
        info["shift_types"] = created_change_types

        # Reconstruct series
        noise = np.random.normal(
            0,
            noise_std,
            size=n
        )

        if seasonal_period is None:
            series = (
                trend
                + residual_component
                + noise
            )

        elif isinstance(seasonal_period, int):
            series = (
                trend
                + seasonal_component
                + residual_component
                + noise
            )

        elif isinstance(seasonal_period, (list, tuple)):
            series = (
                trend
                + seasonal_component.sum(axis=1)
                + residual_component
                + noise
            )

        # Update dataframe
        df.loc[:, "data"] = series
        df.loc[:, "stationary"] = 0

        # Create structural-break regime labels only when requested
        if is_loc is True:
            trend_shift_label = np.zeros(
                n,
                dtype=int
            )

            for regime_number, break_point in enumerate(
                sorted(created_breaks),
                start=1
            ):
                trend_shift_label[break_point:] = regime_number

            df.loc[:, "trend_shift_label"] = trend_shift_label

        return df, info
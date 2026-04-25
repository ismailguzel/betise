import numpy as np
import pandas as pd
import random
from numpy.polynomial import Polynomial
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.filters.hp_filter import hpfilter
import warnings
from arch import arch_model


class TimeSeriesGenerator:
    def __init__(self, length=None):
        self.length = length if length is not None else 400
        self.stationary_base_distributions = ['ar', 'ma', 'arma','white_noise']
        self.seasonal_base_distributions = ['sarma', 'sarima']
        self.volatile_base_distributions = ['arch', 'garch', 'egarch', 'aparch']
        self.stochastic_base_distributions = ['ari', 'ima', 'arima']
        self.characteristics = {'deterministic_trend_linear' : self.generate_deterministic_trend_linear,
        'deterministic_trend_cubic': self.generate_deterministic_trend_cubic,
        'deterministic_trend_quadratic': self.generate_deterministic_trend_quadratic,
        'deterministic_trend_exponential': self.generate_deterministic_trend_exponential,
        'stochastic_trend': self.generate_stochastic_trend,
        'single_seasonality': self.generate_single_seasonality,
        'multiple_seasonality': self.generate_multiple_seasonality,
        'single_point_anomaly' : self.generate_point_anomaly,
        'multiple_point_anomalies': self.generate_point_anomalies,
        'collective_anomalies': self.generate_collective_anomalies,
        'contextual_anomalies': self.generate_contextual_anomalies}
        self.structural_breaks = {'mean_shift': self.generate_mean_shift,
        'variance_shift': self.generate_variance_shift,
        'trend_shift': self.generate_trend_shift}

    #HELPER FUNCTIONS
    
    def z_normalize(self,series):
        return (series - np.mean(series)) / np.std(series)
        
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

    def extract_seasonal_part(self,series, period):
        decomposition = seasonal_decompose(series, model='additive', period=period)
        seasonal = decomposition.seasonal
        return seasonal

    def generate_nonzero_coefs(self, order, low, high, exclusion_lower, exclusion_upper):
        coefs = []
        while len(coefs) < order:
            val = np.random.uniform(low, high)
            if abs(val) >= exclusion_lower and abs(val) <= exclusion_upper:
                coefs.append(val)
        return np.array(coefs)
        

    #BASE DISTRIBUTIONS STATIONARY

    def generate_ar_params(self, order_range=(1, 3), coef_range=(-0.75, 0.75)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = np.random.uniform(coef_range[0], coef_range[1], order)
            ar = np.r_[1, -coefs]
            ma = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if not arma_process.isstationary:
                continue
            # Reject near-unit-root: characteristic roots must be well inside unit circle
            if np.all(np.abs(np.roots(ar)) < 0.98):
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
        series = self.z_normalize(series)
        return series, info
            
    def generate_ma_params(self, order_range=(1, 3), coef_range=(-0.75, 0.75)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = np.random.uniform(coef_range[0], coef_range[1], order)
            ma = np.r_[1, coefs]
            ar = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if not arma_process.isinvertible:
                continue
            # Reject near-unit-root: characteristic roots must be well inside unit circle
            if np.all(np.abs(np.roots(ma)) < 0.98):
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
        series = self.z_normalize(series)
        return series, info

    def generate_arma_params(self, order_range=(1, 3), coef_range=(-0.75, 0.75)):
        while True:
            ar_order = np.random.randint(order_range[0], order_range[1] + 1)
            ma_order = np.random.randint(order_range[0], order_range[1] + 1)
            ar_coefs = np.random.uniform(coef_range[0], coef_range[1], ar_order)
            ma_coefs = np.random.uniform(coef_range[0], coef_range[1], ma_order)
            ma = np.r_[1, ma_coefs]
            ar = np.r_[1, -ar_coefs]
            arma_process = ArmaProcess(ar, ma)
            if not (arma_process.isinvertible and arma_process.isstationary):
                continue
            # Reject near-unit-root: characteristic roots must be well inside unit circle
            if np.all(np.abs(np.roots(ar)) < 0.98) and np.all(np.abs(np.roots(ma)) < 0.98):
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
        series = self.z_normalize(series)
        return series, info

    def generate_white_noise(self, length, noise_std = None):
        info = {'type': 'base_series','subtype': 'white_noise'}
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        series = np.random.normal(0, 1, length)
        series = series + np.random.normal(0,noise_std,length)
        return series, info

    def generate_arima_params(self, order_range=(1, 3), d_range=(1, 2), coef_range = (-0.9,0.9)):
        while True:
            p = np.random.randint(order_range[0], order_range[1] + 1)
            d = np.random.randint(d_range[0], d_range[1] + 1)
            q = np.random.randint(order_range[0], order_range[1] + 1)

            ar_coefs = self.generate_nonzero_coefs(p, coef_range[0], coef_range[1], exclusion_lower=0.2, exclusion_upper=0.8)
            ma_coefs = self.generate_nonzero_coefs(q, coef_range[0], coef_range[1], exclusion_lower=0.2, exclusion_upper=0.8)

            ar = np.r_[1, -ar_coefs]
            ma = np.r_[1, ma_coefs]

            arma_process = ArmaProcess(ar, ma)
            if arma_process.isstationary and arma_process.isinvertible:
                break

        return p, d, q, ar_coefs, ma_coefs

    def generate_arima_series(self, length, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        p, d, q, ar_coefs, ma_coefs = self.generate_arima_params()

        ar = np.r_[1, -ar_coefs]
        ma = np.r_[1, ma_coefs]

        info = {'type': 'trend', 'subtype' : 'stochastic_ARIMA', 'ar_order': p, 'ar_coefs': ar_coefs, 'ma_order': q, 'ma_coefs': ma_coefs, 'diff': d}
        arma_process = ArmaProcess(ar, ma)
        arma_sample = arma_process.generate_sample(nsample=length)

        # Integrate (difference 'd' times)
        series = arma_sample
        for _ in range(d):
            series = np.cumsum(series)

        series = series + np.random.normal(0,noise_std,length)
        series = self.z_normalize(series)
        return series, info

    def generate_ari_params(self, order_range=(1, 3), d_range = (1, 2), coef_range = (-0.9,0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = self.generate_nonzero_coefs(order, coef_range[0], coef_range[1], exclusion_lower = 0.3, exclusion_upper = 0.6)
            d = np.random.randint(d_range[0], d_range[1] + 1)
            ar = np.r_[1, -coefs]
            ma = np.array([1])
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isstationary:
                break
        return d, order, coefs

    def generate_ari_series(self, length, const=False, drift=None, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        d, order, coefs = self.generate_ari_params()
        info = {'type': 'trend', 'subtype' : 'stochastic_ARI', 'ar_order': order, 'ar_coefs': coefs, 'diff': d}
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
        series = self.z_normalize(series)
        return series, info

    def generate_ima_params(self, order_range=(1, 3), d_range = (1,2), coef_range = (-0.9,0.9)):
        while True:
            order = np.random.randint(order_range[0], order_range[1] + 1)
            coefs = self.generate_nonzero_coefs(order, coef_range[0], coef_range[1], exclusion_lower = 0.3, exclusion_upper = 0.6)
            d = np.random.randint(d_range[0], d_range[1] + 1)
            ar = np.array([1])
            ma = np.r_[1, coefs]
            arma_process = ArmaProcess(ar, ma)
            if arma_process.isinvertible:
                break
        return d, order, coefs

    def generate_ima_series(self, length, const=False, drift=None, noise_scale=0.5, noise_std = None):
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        d, order, coefs = self.generate_ima_params()
        info = {'type': 'trend', 'subtype' : 'stochastic_IMA', 'ma_order': order, 'ma_coefs': coefs, 'diff': d}
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
        series = self.z_normalize(series)
        return series, info

    def generate_sarima_params(self, p_range=(1, 3), d_range=(0, 1), q_range=(1, 3), P_range=(1, 3), Q_range=(1, 3), D_range=(0,1), coef_range = (-0.9,0.9)):
        while True:
            p = np.random.randint(p_range[0], p_range[1] + 1)
            d = np.random.randint(d_range[0], d_range[1] + 1)
            q = np.random.randint(q_range[0], q_range[1] + 1)

            if d == 0:
                D = 1
            else:
                D = np.random.randint(D_range[0], D_range[1] + 1)
            
            P = np.random.randint(P_range[0], P_range[1] + 1)
            Q = np.random.randint(Q_range[0], Q_range[1] + 1)
            valid_periods = [s for s in [5, 7, 12, 24, 30, 52, 90, 180] if self.length // 12 <= s <= self.length // 4]
            if not valid_periods:
                continue
            s = random.choice(valid_periods)

            ar_params = self.generate_nonzero_coefs(p, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if p > 0 else np.array([])
            ma_params = self.generate_nonzero_coefs(q, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if q > 0 else np.array([])
            seasonal_ar_params = self.generate_nonzero_coefs(P, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if P > 0 else np.array([])
            seasonal_ma_params = self.generate_nonzero_coefs(Q, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if Q > 0 else np.array([])

            if (self.is_stationary(ar_params) and self.is_invertible(ma_params) and
                self.is_stationary(seasonal_ar_params) and self.is_invertible(seasonal_ma_params)):

                arma_params = np.concatenate([ar_params, ma_params, seasonal_ar_params, seasonal_ma_params])
                return (p, d, q), (P, D, Q, s), arma_params

    def generate_sarima_series(self, length, max_attempts=10, noise_std=None, noise_scale=0.3):
        self.length = length
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        attempts = 0
        while attempts < max_attempts:
            try:
                order, seasonal_order, arma_params = self.generate_sarima_params()
                p, d, q = order
                P, D, Q, s = seasonal_order
                period = s
                warmup = max(8 * s, 200)

                # Skip overly complex models
                if (p + q + P + Q) > 6:
                    continue

                # Skip unstable or uninteresting coefficient sets
                if np.sum(np.abs(arma_params)) < 0.6 or np.max(np.abs(arma_params)) > 0.6:
                    continue

                variance_param = np.array([1.0])
                full_params = np.concatenate([arma_params, variance_param])

                # Use stationary initialization to avoid diffuse Kalman filter transient
                endog_dummy = np.zeros(length + warmup)
                model = SARIMAX(
                    endog=endog_dummy,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse')

                series = model.simulate(params=full_params, nsimulations=length + warmup,
                                        initial_state=np.zeros(model.k_states))
                series = series[warmup:]

                if (np.std(series[-length//2:]) < 0.05 or np.max(np.abs(series)) < 0.3):
                    print("Flat or decaying series — discarded")
                    print(f"Order: {order}, Seasonal Order: {seasonal_order}")
                    print("Coefficients:", arma_params)
                    continue

                # Reject series whose first 15% has >3x higher std than the stable middle
                head_std   = np.std(series[:length // 7])
                middle_std = np.std(series[length // 4: 3 * length // 4])
                if middle_std > 0 and head_std > 3.0 * middle_std:
                    continue

                series += np.random.normal(0, noise_std * 0.2, length)
                series = self.z_normalize(series)
                info = {'type': 'seasonal', 'subtype': 'SARIMA', 'period': period, 'ar_order':p, 'ma_order':q, 'diff':d, 'seasonal_ar_order':P, 'seasonal_ma_order': Q, 'seasonal_diff': D, 'coefs': arma_params}
                return series, info

            except (ValueError, np.linalg.LinAlgError):
                attempts += 1
                print(f"Attempt {attempts}/{max_attempts} failed. Retrying...")

        print("SARIMA generation failed. Returning None.")
        return None, None  # Return None on failure

    def generate_sarma_params(self, p_range=(1, 3), q_range=(1, 3), P_range=(1, 3), Q_range=(1, 3), coef_range = (-0.9,0.9)):
        while True:
            p = np.random.randint(p_range[0], p_range[1] + 1)
            q = np.random.randint(q_range[0], q_range[1] + 1)
            d = 0

            P = np.random.randint(P_range[0], P_range[1] + 1)
            Q = np.random.randint(Q_range[0], Q_range[1] + 1)
            D = 0
            valid_periods = [s for s in [5, 7, 12, 24, 30, 52, 90, 180] if self.length // 12 <= s <= self.length // 4]
            if not valid_periods:
                continue
            s = random.choice(valid_periods)

            ar_params = self.generate_nonzero_coefs(p, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if p > 0 else np.array([])
            ma_params = self.generate_nonzero_coefs(q, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if q > 0 else np.array([])
            seasonal_ar_params = self.generate_nonzero_coefs(P, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if P > 0 else np.array([])
            seasonal_ma_params = self.generate_nonzero_coefs(Q, coef_range[0], coef_range[1], exclusion_lower=0.3, exclusion_upper=0.6) if Q > 0 else np.array([])

            if (self.is_stationary(ar_params) and self.is_invertible(ma_params) and
                self.is_stationary(seasonal_ar_params) and self.is_invertible(seasonal_ma_params)):

                arma_params = np.concatenate([ar_params, ma_params, seasonal_ar_params, seasonal_ma_params])
                return (p, d, q), (P, D, Q, s), arma_params

    def generate_sarma_series(self, length, max_attempts=10, noise_std=None, noise_scale=0.3):
        self.length = length
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
        attempts = 0
        while attempts < max_attempts:
            try:
                order, seasonal_order, arma_params = self.generate_sarma_params()
                p, d, q = order
                P, D, Q, s = seasonal_order
                period = s
                warmup = max(8 * s, 200)

                # Skip overly complex models
                if (p + q + P + Q) > 6:
                    continue

                # Skip unstable or uninteresting coefficient sets
                if np.sum(np.abs(arma_params)) < 0.6 or np.max(np.abs(arma_params)) > 0.6:
                    continue

                variance_param = np.array([1.0])
                full_params = np.concatenate([arma_params, variance_param])

                # Use stationary initialization to avoid diffuse Kalman filter transient
                endog_dummy = np.zeros(length + warmup)
                model = SARIMAX(
                    endog=endog_dummy,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse')

                series = model.simulate(params=full_params, nsimulations=length + warmup,
                                        initial_state=np.zeros(model.k_states))
                series = series[warmup:]

                # Post-filter flat or unrealistic series
                if (np.std(series[-length//2:]) < 0.05 or np.max(np.abs(series)) < 0.3):
                    print("Flat or decaying series — discarded")
                    print(f"Order: {order}, Seasonal Order: {seasonal_order}")
                    print("Coefficients:", arma_params)
                    continue

                # Reject series whose first 15% has >3x higher std than the stable middle
                head_std   = np.std(series[:length // 7])
                middle_std = np.std(series[length // 4: 3 * length // 4])
                if middle_std > 0 and head_std > 3.0 * middle_std:
                    continue

                series += np.random.normal(0, noise_std * 0.2, length)
                series = self.z_normalize(series)
                info = {'type': 'seasonal', 'subtype': 'SARMA', 'period': period, 'ar_order':p, 'ma_order':q, 'diff':d, 'seasonal_ar_order':P, 'seasonal_ma_order': Q, 'seasonal_diff': D, 'coefs': arma_params}
                return series, info

            except (ValueError, np.linalg.LinAlgError):
                attempts += 1
                print(f"Attempt {attempts}/{max_attempts} failed. Retrying...")

        print("SARMA generation failed. Returning None.")
        return None, None  # Return None on failure

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

    def generate_point_anomaly(self, df, location=None, scale_factor=0.5, is_spike=True):
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
            global_spike_factor = np.random.uniform(1.1,1.2) #
            if is_spike:
                magnitude = global_spike_factor * global_spike * scale_factor
            else:
                magnitude = local_std * np.random.uniform(1.5, 2.0) * scale_factor
            direction = np.random.choice([-1, 1])
            series[idx] = np.mean(series) + direction * magnitude

        info = {'type': 'anomaly', 'subtype': 'single_point', 'num_anomalies': num_anomalies, 'anomaly_indices': anomaly_indices, 'location': location}
    
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'point_anom_single'] = 1
        return df, info

    def generate_point_anomalies(self, df, scale_factor=0.5):
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
        return df, info


    def generate_collective_anomalies(self, df, num_anomalies=1, location=None, scale_factor=1):
        series = df['data'].copy()
        n = len(series)
        min_distance = int(0.1 * n)
        ends = []

        if num_anomalies == 1:
            # Determine candidate start regions based on location
            if location == "beginning":
                candidate_range = np.arange(int(0.1 * n), int(0.3 * n))
            elif location == "middle":
                candidate_range = np.arange(int(0.4 * n), int(0.6 * n))
            elif location == "end":
                candidate_range = np.arange(int(0.7 * n), int(0.9 * n))
            else:  # "random" or None — full safe zone
                candidate_range = np.arange(int(0.1 * n), int(0.9 * n))
        else:
            candidate_range = np.arange(int(0.1 * n), int(0.85 * n))  # Safe zone
            location = 'none'
            
        # Select non-overlapping anomaly start points
        selected_starts = []
        candidates = candidate_range.copy()
        while len(selected_starts) < num_anomalies and len(candidates) > 0:
            start = np.random.choice(candidates)
            selected_starts.append(start)
            candidates = candidates[np.abs(candidates - start) >= min_distance]
    
        # Apply collective anomalies with controlled magnitude and internal noise
        for start in selected_starts:
            max_length = n - start - int(n * 0.05)  # Ensure it doesn't exceed series end
            anomaly_length = np.random.randint(int(n * 0.05), min(int(n * 0.09), max_length))
            end = min(n, start + anomaly_length)
            ends.append(end)
    
            local_std = np.std(series)
            magnitude = np.random.uniform(0.8, 1.5) * local_std * scale_factor
            shift = np.random.choice([-1, 1]) * magnitude
            #####internal_noise = np.random.normal(0, 0.1 * local_std, end - start)#####
    
            series[start:end] += shift

        info = {'type': 'anomaly', 'subtype': 'collective','num_anomalies': num_anomalies, 'location': location}
    
        # Sort and update DataFrame
        selected_starts = np.sort(selected_starts)
        ends = np.sort(ends)
        info['starts'] = selected_starts
        info['ends'] = ends
        
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'collect_anom'] = 1
        return df, info

    def generate_contextual_anomalies(self, df, num_anomalies=1, location=None, scale_factor=1,
                                  anomaly_strength=1, seasonal_period=None, max_attempts=10):    
        series_original = df['data'].copy()
        n = len(series_original)
        info = []

        for attempt in range(max_attempts):
            min_distance = max(1, int((0.05 - attempt * 0.003) * n))  # gradually relax
            series = series_original.copy()
            selected_starts = []
            ends = []

            # Decide the seasonal period
            if seasonal_period is not None:
                period = seasonal_period
                generate_seasonality = False
            else:
                min_period = max(5, n // 20)  # slightly more lenient lower bound
                max_period = n // 6
                allowed = [5, 7, 12, 24, 30, 52, 90, 180]
                periods = [p for p in allowed if min_period <= p <= max_period]
                if not periods:
                    continue  # try again
                period = random.choice(periods)
                generate_seasonality = True

            # Generate or estimate seasonality
            if generate_seasonality:
                amplitude = np.std(series) * np.random.uniform(1.5, 3)
                seasonality = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
                series += seasonality * scale_factor
            else:
                seasonality = np.sin(2 * np.pi * np.arange(n) / period)

            # Find contextual points from clean sine wave
            pure_seasonality = np.sin(2 * np.pi * np.arange(n) / period)
            peaks = np.where((pure_seasonality[1:-1] > pure_seasonality[:-2]) &
                         (pure_seasonality[1:-1] > pure_seasonality[2:]))[0] + 1
            valleys = np.where((pure_seasonality[1:-1] < pure_seasonality[:-2]) &
                           (pure_seasonality[1:-1] < pure_seasonality[2:]))[0] + 1
            candidate_indices = np.concatenate([peaks, valleys])

            # Determine candidate regions
            if num_anomalies == 1:
                if location == "beginning":
                    candidate_range = np.arange(int(0.1 * n), int(0.3 * n))
                elif location == "middle":
                    candidate_range = np.arange(int(0.4 * n), int(0.6 * n))
                elif location == "end":
                    candidate_range = np.arange(int(0.7 * n), int(0.9 * n))
                else:
                    candidate_range = np.arange(int(0.1 * n), int(0.85 * n))
                    location = 'none'
            else:
                candidate_range = np.arange(int(0.1 * n), int(0.85 * n))
                location = 'none'

            candidate_indices = np.array([i for i in candidate_indices if i in candidate_range])

            if len(candidate_indices) == 0:
                print(f"[Attempt {attempt+1}] No candidates found for n={n}, period={period}")
                continue

            # Try to select num_anomalies with spacing
            candidates = candidate_indices.copy()
            np.random.shuffle(candidates)
            for center in candidates:
                if all(abs(center - prev) >= min_distance for prev in selected_starts):
                    selected_starts.append(center)
                if len(selected_starts) == num_anomalies:
                    break

            # If still not enough, just fill the rest from remaining candidates (ignore spacing)
            if len(selected_starts) < num_anomalies:
                remaining = list(set(candidate_indices) - set(selected_starts))
                np.random.shuffle(remaining)
                for center in remaining:
                    selected_starts.append(center)
                    if len(selected_starts) == num_anomalies:
                        break

            if len(selected_starts) == 0:
                continue

            # Apply contextual anomalies
            for center in selected_starts:
                anomaly_length = min(max(int(period * 0.5), 10), int(0.2 * n))  # safe max
                start = max(0, center - anomaly_length // 2)
                end = min(n, start + anomaly_length)
                ends.append(end)

                local_season = seasonality[start:end]
                series[start:end] -= 2 * local_season * anomaly_strength  # FLIP!

            # Success — break retry loop
            break
        else:
            # Tüm denemeler başarısız olduysa, orijinal df'i ve None info'yu döndür
            print(f"generate_contextual_anomalies failed for n={n}")
            return df, None 

        info = {'type': 'anomaly', 'subtype': 'contextual','num_anomalies': num_anomalies, 'location': location}
        
        # Labeling
        selected_starts = np.sort(selected_starts)
        ends = np.sort(ends)
        info['starts'] = selected_starts
        info['ends'] = ends
        
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'context_anom'] = 1
        df.loc[:, 'seasonal'] = 1
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
        series = self.z_normalize(series)
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
        series = self.z_normalize(series)

        info = {'type' : 'trend', 'subtype': 'deterministic_cubic','sign': sign, 'a': a, 'b': b}
        
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'det_cubic'] = 1
        return df, info

    def generate_deterministic_trend_exponential(self, df, sign=None, a=None, b=None, noise_std=None, scale_factor=1):
        series = df['data'].copy()
        sign = sign if sign in [-1, 1] else random.choice([-1, 1])
        length = len(series)
        a = a if a is not None else random.uniform(0.5, 1.5)
        b = b if b is not None else random.uniform(1.8, 2.8)
        t = np.linspace(0, 1, len(series))

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
    
        series = trend + noise*3

        series = self.z_normalize(series)

        info = {'type' : 'trend', 'subtype': 'deterministic_exponential','sign': sign, 'a': a, 'b': b}
    
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        df.loc[:, 'det_exp'] = 1
        return df, info

    #TRENDS - STOCHASTIC TRENDS

    def generate_stochastic_trend(self, kind='rw', const=False, drift=None, noise_std=1.0):
        t = np.arange(self.length)
        noise = np.random.normal(0, noise_std, self.length)
    
        if kind == 'rw':
            info = {'type': 'trend', 'subtytpe': 'random_walk'}
            series = np.cumsum(noise)
            series = self.z_normalize(series)
    
        elif kind == 'rwd':
            if drift is None:
                drift = np.random.uniform(0.01, 0.1)
            info = {'type': 'trend', 'subtype': 'random_walk_with_drift', 'drift': drift}
            series = drift * t + np.cumsum(noise)
            series = self.z_normalize(series)
    
        elif kind == 'ari':
            series, info = self.generate_ari_series(length=self.length)
    
        elif kind == 'ima':
            series, info = self.generate_ima_series(length=self.length)
    
        elif kind == 'arima':
            series, info = self.generate_arima_series(length=self.length)
    
        else:
            raise ValueError("Invalid kind. Choose from 'rw', 'rwd', 'ari', 'ima', or 'arima'.")

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': (np.zeros(self.length)).astype(int),
            'seasonal': np.zeros(self.length).astype(int),
        })
        return df, info

    #SEASONALITY

    def generate_single_seasonality(self, df, period=None, amplitude=None, noise_std=None, scale_factor = 3):
        series = np.random.normal(loc=0.0, scale=0.2, size=self.length)
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        min_period = 5  
        max_period = len(series) // 6  # Ensure at least 6 cycles
        periods = [p for p in [5, 7, 12, 24, 30, 52, 90, 180] if min_period <= p <= max_period]
        period = period if period is not None else random.choice(periods)
        amplitude = amplitude if amplitude is not None else np.std(series) * np.random.uniform(0.5, 2.5)
        seasonality = (amplitude * np.sin(2 * np.pi * np.arange(len(series)) / period) + np.random.normal(0, noise_std, size = len(series)))
        series += seasonality * scale_factor
        series = self.z_normalize(series)
        info = {'type': 'seasonal', 'subtype': 'single_seasonality', 'period': period, 'amplitude': amplitude}
        df.loc[:,'data'] = series
        df.loc[:,'stationary'] = 0
        df.loc[:, 'seasonal'] = 1
        df.loc[:,'single_seas'] = 1
        return df, info

    def generate_multiple_seasonality(self, df, num_components=2, periods=None, amplitudes=None, noise_std=None, scale_factor=3):
        series = np.random.normal(loc=0.0, scale=0.2, size=self.length)
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        info = {'type': 'seasonal', 'subtype': 'multiple_seasonality'}
        min_period = 5  
        max_period = len(series) // 6
        valid_periods = [p for p in [5, 7, 12, 24, 30, 52, 90, 180] if min_period <= p <= max_period]
        periods_meta = []
        amplitudes_meta = []

        if periods is None:
            periods = random.sample(valid_periods, min(num_components, len(valid_periods)))

        if amplitudes is None:
            base_std = np.std(series)
            amplitudes = [base_std * np.random.uniform(0.5, 2.0) for _ in periods]

        for i, (period, amplitude) in enumerate(zip(periods, amplitudes), start = 1 ):
            seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(len(series)) / period)
            seasonal_component += np.random.normal(0, noise_std, size=len(series))
            series += seasonal_component * scale_factor
            series = self.z_normalize(series)
            periods_meta.append(period)
            amplitudes_meta.append(amplitude)

        info['periods'] = periods_meta
        info['amplitudes'] = amplitudes_meta

        df.loc[:, 'data'] = series
        df.loc[:, 'multiple_seas'] = 1
        df.loc[:,'stationary'] = 0
        df.loc[:, 'seasonal'] = 1
        return df, info


    def generate_seasonality_from_base_series(self, kind = None, num_components = 2):
        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': np.ones(self.length),
            'stationary': (np.zeros(self.length)).astype(int)
        })

        if kind == 'single':
            df, info = self.generate_single_seasonality(df)
        if kind == 'multiple':
            df, info = self.generate_multiple_seasonality(df = df, num_components = num_components)
        if kind == 'sarma':
            series, info = self.generate_sarma_series(self.length)
            if series is None: return None, None  # Return None if generation failed
            series = self.z_normalize(series)
            df.loc[:, 'data'] = series
            df.loc[:, 'seasonal_base'] = 1
            df.loc[:, 'seasonal'] = 1
        if kind == 'sarima':
            series, info = self.generate_sarima_series(self.length)
            if series is None: return None, None  # Return None if generation failed
            series = self.z_normalize(series)
            df.loc[:, 'data'] = series
            df.loc[:, 'seasonal_base'] = 1
            df.loc[:, 'seasonal'] = 1

        return df, info

    #VOLATILITY

    def generate_volatility(self, kind = None):
        if kind == 'arch':
            series, info = self.generate_arch_series(self.length)
            series = self.z_normalize(series)
        elif kind == 'garch':
            series, info = self.generate_garch_series(self.length)
            series = self.z_normalize(series)
        elif kind == 'egarch':
            series, info = self.generate_egarch_series(self.length)
            series = self.z_normalize(series)
        elif kind == 'aparch':
            series, info = self.generate_aparch_series(self.length)
            series = self.z_normalize(series)

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': (np.zeros(self.length)).astype(int),
            'seasonal': np.zeros(self.length).astype(int),
        })
        return df, info


    #STRUCTURAL BREAKS
    
    def generate_mean_shift(self, df, signs=None, location=None, num_breaks=None, noise_std = None, scale_factor=1, seasonal_period=None):
        series = df['data'].copy()
        n = len(series)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            residuals, trend = hpfilter(series, lamb=11000)
        noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
        min_distance = 0.1 * n
        created_breaks = []
        magnitudes = []
        info = []
        
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
                if seasonal_period:
                    phase = point % seasonal_period
                    point -= phase
                break_points.append(point)
                candidates = candidates[np.abs(candidates - point) >= min_distance]
            break_points = sorted(break_points)

        info = {'type': 'structural_break', 'subtype': 'mean_shift', 'num_breaks':num_breaks, 'location' : location}
        
        # Apply shifts
        for i, break_point in enumerate(break_points):
            magnitude = np.random.uniform(0.3, 1.5) #0.2-1
            magnitudes.append(magnitude)
            level_shift = signs[i] * magnitude
    
            trend[break_point:] += level_shift * scale_factor 
            created_breaks.append(break_point)

        info['shift_indices'] = created_breaks
        info['shift_magnitudes'] = magnitudes
    
        # Reconstruct series
        noise = np.random.normal(0, noise_std, n)
        series = residuals + trend + noise
        series = self.z_normalize(series)
        df.loc[:,'data'] = series
        df.loc[:,'stationary'] = 0
        return df, info


    def generate_variance_shift(self, df, signs=None, location=None, num_breaks=None, scale_factor=1, seasonal_period=None):
        series = df['data'].copy()
        n = len(series)
        min_distance = 0.1 * n
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            residuals, trend = hpfilter(series, lamb=11000)
        created_breaks = []
        variance_change_factors = []

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
                if seasonal_period:
                    phase = point % seasonal_period
                    point -= phase
                break_points.append(point)
                candidates = candidates[np.abs(candidates - point) >= min_distance]
            break_points = sorted(break_points)

        info = {'type': 'structural_break', 'subtype': 'variance_shift', 'num_breaks':num_breaks, 'location' : location}
        
        for i, break_point in enumerate(break_points):
            variance_factor = np.random.uniform(1.5, 3)
            variance_change_factors.append(variance_factor)
            if signs[i] > 0:
                residuals[break_point:] *= variance_factor * scale_factor
            elif signs[i] < 0:
                residuals[break_point:] /= variance_factor * scale_factor
    
            created_breaks.append((break_point))

        info['shift_indices'] = created_breaks
        info['shift_magnitudes'] = variance_change_factors
    
        series = residuals + trend
        series = self.z_normalize(series)
    
        df.loc[:,'data'] = series
        df.loc[:,'stationary'] = 0
        return df, info


    def generate_trend_shift(self, df, sign=None, location=None, num_breaks=None, change_types=None,
                             seasonal_period=None, scale_factor=1, noise_std=None):
        series = df['data'].copy()
        n = len(series)
        min_distance = 0.1 * n
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            residuals, trend = hpfilter(series, lamb=11000)
        noise_std = noise_std if noise_std is not None else np.random.uniform(5, 15)
        created_breaks = []
        created_change_types = []
    
        if seasonal_period:
            seasonality = np.sin(2 * np.pi * np.arange(n) / seasonal_period)
        else:
            seasonality = np.zeros(n)
    
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
                if seasonal_period:
                    phase = point % seasonal_period
                    point -= phase
                break_points.append(point)
                candidates = candidates[np.abs(candidates - point) >= min_distance]
            break_points = sorted(break_points)
    
        # Validate change_types input
        if change_types is None or len(change_types) != len(break_points):
            raise ValueError("change_types must be a list with the same length as the number of breaks.")
    
        # Initialize trend array
        current_slope = np.random.uniform(0.1, 0.9) * sign
        current_level = 0
        prev_point = 0

        info = {'type': 'structural_break', 'subtypse': 'trend_shift', 'num_breaks': num_breaks, 'location' :location}
        
        for i, break_point in enumerate(break_points + [n]):  # Include end of series
            slope_change_factor = np.random.uniform(1.5, 4.5)
            segment_length = break_point - prev_point
            segment_trend = current_level + current_slope * np.arange(segment_length)
            trend[prev_point:break_point] = segment_trend
            current_level = segment_trend[-1]
    
            if break_point == n:
                break
    
            change_type = change_types[i]
    
            if change_type == 'direction_change':
                current_slope = -current_slope
            elif change_type == 'magnitude_change':
                current_slope = current_slope * slope_change_factor
            elif change_type == 'direction_and_magnitude_change':
                current_slope = -current_slope * slope_change_factor
            else:
                raise ValueError("Invalid change_type: " + str(change_type))
    
            created_breaks.append(break_point)
            created_change_types.append(change_type)
            prev_point = break_point

        info['shift_indices'] = created_breaks
        info['shift_types'] = created_change_types
    
        # Final series
        seasonal_factor = np.random.uniform(30, 45)
        series = trend + residuals + seasonality * seasonal_factor + np.random.normal(0, noise_std, size=n)
        series = self.z_normalize(series)
    
        # Update dataframe
        df.loc[:, 'data'] = series
        df.loc[:, 'stationary'] = 0
        return df, info


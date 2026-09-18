import numpy as np
import pandas as pd
import random
from numpy.polynomial import Polynomial
from statsmodels.tsa.arima_process import ArmaProcess
from arch import arch_model
from statsmodels.tsa.seasonal import STL,MSTL
from betise.utils.arfima_simulator import ARFIMA_sim
from scipy.signal import fftconvolve, lfilter

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
        'pure_sarima_seasonality': self.generate_pure_sarima,
        'pure_sarma_seasonality': self.generate_pure_sarma,
        'deterministic_sarma_seasonality': self.generate_deterministic_sarma,
        'deterministic_sarima_seasonality': self.generate_deterministic_sarima,
        'seasonal_unit_root_fourier': self.generate_seasonal_unit_root_fourier,
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

    def generate_ar_series(
        self,
        length,
        noise_std=None,
        innovations=None
    ):
        order, coefs = self.generate_ar_params()

        info = {
            'type': 'base_series',
            'subtype': 'AR',
            'ar_order': order,
            'ar_coefs': coefs
        }

        ar = np.r_[1, -np.array(coefs)]
        ma = np.array([1])

        ar_process = ArmaProcess(ar, ma)

        # Standard AR:
        # use Gaussian innovations generated internally.
        if innovations is None:
            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            series = ar_process.generate_sample(
                nsample=length,
                scale=noise_std
            )

        # AR + another innovation mechanism
        # e.g. AR + GARCH
        else:
            innovations = np.asarray(innovations)

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            series = ar_process.generate_sample(
                nsample=length,
                scale=1.0,
                distrvs=lambda size: innovations
            )

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

    def generate_ma_series(
        self,
        length,
        noise_std=None,
        innovations=None
    ):
        order, coefs = self.generate_ma_params()

        info = {
            'type': 'base_series',
            'subtype': 'MA',
            'ma_order': order,
            'ma_coefs': coefs
        }

        ar = np.array([1])
        ma = np.r_[1, np.array(coefs)]

        ma_process = ArmaProcess(ar, ma)

        # Standard MA:
        # Gaussian innovations are generated internally.
        if innovations is None:
            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            series = ma_process.generate_sample(
                nsample=length,
                scale=noise_std
            )

        # MA + another innovation mechanism
        # e.g. MA + GARCH
        else:
            innovations = np.asarray(innovations)

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            series = ma_process.generate_sample(
                nsample=length,
                scale=1.0,
                distrvs=lambda size: innovations
            )

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

    def generate_arma_series(
        self,
        length,
        noise_std=None,
        innovations=None
    ):
        ar_order, ma_order, ar_coefs, ma_coefs = self.generate_arma_params()

        info = {
            'type': 'base_series',
            'subtype': 'ARMA',
            'ar_order': ar_order,
            'ar_coefs': ar_coefs,
            'ma_order': ma_order,
            'ma_coefs': ma_coefs
        }

        ar = np.r_[1, -np.array(ar_coefs)]
        ma = np.r_[1, np.array(ma_coefs)]

        arma_process = ArmaProcess(ar, ma)

        # Standard ARMA:
        # Gaussian innovations are generated internally.
        if innovations is None:
            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            series = arma_process.generate_sample(
                nsample=length,
                scale=noise_std
            )

        # ARMA + another innovation mechanism
        # e.g. ARMA + GARCH
        else:
            innovations = np.asarray(innovations)

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            series = arma_process.generate_sample(
                nsample=length,
                scale=1.0,
                distrvs=lambda size: innovations
            )

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

    def generate_arima_series(
        self,
        length,
        d=1,
        const=False,
        drift=None,
        noise_std=None,
        innovations=None
    ):
        if d not in {1, 2}:
            raise ValueError(
                "ARIMA differencing order d must be 1 or 2."
            )

        p, q, ar_coefs, ma_coefs = self.generate_arima_params()

        ar = np.r_[1, -ar_coefs]
        ma = np.r_[1, ma_coefs]

        unit_root_label = ("1_unit_root" if d == 1 else "2_unit_root")

        arma_process = ArmaProcess(
            ar,
            ma
        )

        # -----------------------------------------------------
        # Standalone ARIMA:
        # Gaussian innovations generated internally
        # -----------------------------------------------------
        if innovations is None:

            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            arma_sample = (
                arma_process.generate_sample(
                    nsample=length,
                    scale=noise_std
                )
            )

        # -----------------------------------------------------
        # ARIMA + external innovation process
        # e.g. ARIMA + GARCH
        # -----------------------------------------------------
        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            arma_sample = (
                arma_process.generate_sample(
                    nsample=length,
                    scale=1.0,
                    distrvs=lambda size: innovations
                )
            )

        # -----------------------------------------------------
        # Integrate d times
        # -----------------------------------------------------
        series = arma_sample.copy()

        for _ in range(d):
            series = np.cumsum(series)

        # -----------------------------------------------------
        # Optional deterministic drift
        # -----------------------------------------------------
        if const:

            if drift is None:
                drift = np.random.uniform(
                    0.01,
                    0.08
                )

            series = (
                series
                + drift * np.arange(length)
            )

        info = {
            'type': 'trend',
            'subtype': 'stochastic_ARIMA',
            'unit_root': unit_root_label,
            'ar_order': p,
            'ar_coefs': ar_coefs,
            'ma_order': q,
            'ma_coefs': ma_coefs,
            'diff': d,
            'drift': drift if const else None
        }

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

    def generate_ari_series(
        self,
        length,
        d=1,
        const=False,
        drift=None,
        noise_std=None,
        innovations=None
    ):
        if d not in {1, 2}:
            raise ValueError(
                "ARI differencing order d must be 1 or 2."
            )

        order, coefs = self.generate_ari_params()

        ar = np.r_[1, -coefs]
        ma = np.array([1])

        unit_root_label = ("1_unit_root" if d == 1 else "2_unit_root")

        ar_process = ArmaProcess(
            ar,
            ma
        )

        # -----------------------------------------------------
        # Standalone ARI
        # -----------------------------------------------------
        if innovations is None:

            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            stationary_sample = (
                ar_process.generate_sample(
                    nsample=length,
                    scale=noise_std
                )
            )

        # -----------------------------------------------------
        # ARI + external innovations
        # -----------------------------------------------------
        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            stationary_sample = (
                ar_process.generate_sample(
                    nsample=length,
                    scale=1.0,
                    distrvs=lambda size: innovations
                )
            )

        # -----------------------------------------------------
        # Integrate d times
        # -----------------------------------------------------
        series = stationary_sample.copy()

        for _ in range(d):
            series = np.cumsum(series)

        # -----------------------------------------------------
        # Optional deterministic drift
        # -----------------------------------------------------
        if const:

            if drift is None:
                drift = np.random.uniform(
                    0.01,
                    0.08
                )

            series = (
                series
                + drift * np.arange(length)
            )

        info = {
            'type': 'trend',
            'subtype': 'stochastic_ARI',
            'unit_root': unit_root_label,
            'ar_order': order,
            'ar_coefs': coefs,
            'diff': d,
            'drift': drift if const else None
        }

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

    def generate_ima_series(
        self,
        length,
        d=1,
        const=False,
        drift=None,
        noise_std=None,
        innovations=None
    ):
        if d not in {1, 2}:
            raise ValueError(
                "IMA differencing order d must be 1 or 2."
            )

        order, coefs = self.generate_ima_params()

        ar = np.array([1])
        ma = np.r_[1, coefs]

        unit_root_label = ("1_unit_root" if d == 1 else "2_unit_root")

        ma_process = ArmaProcess(
            ar,
            ma
        )

        # -----------------------------------------------------
        # Standalone IMA
        # -----------------------------------------------------
        if innovations is None:

            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            stationary_sample = (
                ma_process.generate_sample(
                    nsample=length,
                    scale=noise_std
                )
            )

        # -----------------------------------------------------
        # IMA + external innovations
        # -----------------------------------------------------
        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {length}, got {len(innovations)}."
                )

            stationary_sample = (
                ma_process.generate_sample(
                    nsample=length,
                    scale=1.0,
                    distrvs=lambda size: innovations
                )
            )

        # -----------------------------------------------------
        # Integrate d times
        # -----------------------------------------------------
        series = stationary_sample.copy()

        for _ in range(d):
            series = np.cumsum(series)

        # -----------------------------------------------------
        # Optional deterministic drift
        # -----------------------------------------------------
        if const:

            if drift is None:
                drift = np.random.uniform(
                    0.01,
                    0.08
                )

            series = (
                series
                + drift * np.arange(length)
            )

        info = {
            'type': 'trend',
            'subtype': 'stochastic_IMA',
            'unit_root': unit_root_label,
            'ma_order': order,
            'ma_coefs': coefs,
            'diff': d,
            'drift': drift if const else None
        }

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
        numseas=100,
        innovations=None,
    ):
        """
        Generate ARFIMA(p, d, q).

        Standalone ARFIMA
        -----------------
        Uses the existing Davies-Harte based ARFIMA_sim.

        ARFIMA + external innovation process
        ------------------------------------
        If innovations are supplied:

            Phi(B)(1-B)^d X_t = Theta(B) epsilon_t

        where epsilon_t may come from ARCH/GARCH/EGARCH/APARCH.

        The external innovation path uses the fractional integration filter

            (1-B)^(-d)

        followed by the MA and AR filters.
        """

        p, d, q, ar_coefs, ma_coefs = self.generate_arfima_params(
            d_range=d_range
        )

        external_innovations_used = innovations is not None

        # =====================================================
        # STANDALONE ARFIMA
        # Existing exact Gaussian Davies-Harte implementation
        # =====================================================

        if innovations is None:
            noise_std = (
                noise_std
                if noise_std is not None
                else np.random.uniform(0.1, 1.5)
            )

            series = ARFIMA_sim(
                p_coeffs=ar_coefs,
                q_coeffs=ma_coefs,
                d=d,
                slen=length,
                alpha=alpha,
                sigma=noise_std,
                numseas=numseas,
            )

        # =====================================================
        # ARFIMA + EXTERNAL INNOVATIONS
        # =====================================================

        else:
            innovations = np.asarray(
                innovations,
                dtype=float
            )

            expected_length = length + numseas

            if len(innovations) != expected_length:
                raise ValueError(
                    "ARFIMA external innovations must include the "
                    f"burn-in period. Expected {expected_length} values "
                    f"(length={length} + numseas={numseas}), "
                    f"got {len(innovations)}."
                )

            total = len(innovations)

            # -------------------------------------------------
            # Fractional integration weights
            #
            # (1-B)^(-d)
            #
            # psi_0 = 1
            # psi_k = psi_(k-1) * (k - 1 + d) / k
            # -------------------------------------------------

            weights = np.empty(
                total,
                dtype=float
            )

            weights[0] = 1.0

            for k in range(1, total):
                weights[k] = (
                    weights[k - 1]
                    * (k - 1 + d)
                    / k
                )

            fractional_noise = fftconvolve(
                innovations,
                weights,
                mode="full"
            )[:total]

            # -------------------------------------------------
            # ARMA filtering
            #
            # Phi(B) X_t = Theta(B) W_t
            # -------------------------------------------------

            ar_poly = np.r_[
                1.0,
                -np.asarray(ar_coefs, dtype=float)
            ]

            ma_poly = np.r_[
                1.0,
                np.asarray(ma_coefs, dtype=float)
            ]

            filtered = lfilter(
                ma_poly,
                ar_poly,
                fractional_noise
            )

            # Remove start-up / fractional-filter transient.
            series = filtered[numseas:numseas + length]

            series = alpha + series

            # sigma is not independently generated in this path.
            noise_std = None

        info = {
            "type": "base_series",
            "subtype": "ARFIMA",
            "p": p,
            "d": d,
            "q": q,
            "ar_order": p,
            "ar_coefs": ar_coefs,
            "ma_order": q,
            "ma_coefs": ma_coefs,
            "diff": d,
            "stationary": d < 0.5,
            "fractionally_integrated": True,
            "long_memory": 0 < d < 0.5,
            "alpha": alpha,
            "sigma": noise_std,
            "numseas": numseas,
            "external_innovations_used": external_innovations_used,
        }

        return series, info

    def generate_fractional_process(
        self,
        kind=None,
        d_range=(0.25, 0.49),
        noise_std=None,
        alpha=0,
        numseas=100,
        innovations= None
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
                numseas=numseas,
                innovations=innovations
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

    def generate_egarch_series(
        self,
        length,
        omega_range=(0.1, 0.3),
        alpha_range=(0.2, 0.4),
        gamma_range=(-0.3, 0.3),
        beta_range=(0.75, 0.9),
        cumulative=False,
        scale_factor=1
    ):
        omega = np.random.uniform(*omega_range)
        alpha = np.random.uniform(*alpha_range)
        gamma = np.random.uniform(*gamma_range)
        beta = np.random.uniform(*beta_range)

        am = arch_model(
            None,
            vol='EGARCH',
            p=1,
            o=1,
            q=1,
            mean='Zero',
            dist='normal'
        )

        sim = am.simulate(
            [omega, alpha, gamma, beta],
            nobs=length
        )

        series = sim['data'].values * scale_factor

        info = {
            'type': 'volatility',
            'subtype': 'EGARCH',
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta
        }

        if cumulative:
            series = np.cumsum(series)

        return series, info


    def generate_aparch_series(
        self,
        length,
        omega_range=(0.1, 0.3),
        alpha_range=(0.2, 0.35),
        beta_range=(0.5, 0.75),
        gamma_range=(-0.3, 0.3),
        delta_range=(1.0, 2.0),
        cumulative=False,
        scale_factor=1
    ):
        # Practical stability guard for generated parameters.
        while True:
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)

            if alpha + beta < 1:
                break

        omega = np.random.uniform(*omega_range)
        gamma = np.random.uniform(*gamma_range)
        delta = np.random.uniform(*delta_range)

        am = arch_model(
            None,
            vol='APARCH',
            p=1,
            o=1,
            q=1,
            mean='Zero',
            dist='normal'
        )

        sim = am.simulate(
            [omega, alpha, gamma, beta, delta],
            nobs=length
        )

        series = sim['data'].values * scale_factor

        info = {
            'type': 'volatility',
            'subtype': 'APARCH',
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'delta': delta
        }

        if cumulative:
            series = np.cumsum(series)

        return series, info

    def generate_volatility(
        self,
        kind=None,
        as_innovations=False):
        """
        Generate a volatility process.

        Parameters
        ----------
        kind : str, optional
            One of: arch, garch, egarch, aparch.
            If None, one is selected randomly.

        as_innovations : bool
            If True, return the generated volatility series directly
            so that it can be used as the innovation process of another
            base model.

            If False, return a standalone volatility DataFrame.
        """

        if kind is None:
            kind = np.random.choice(self.volatile_base_distributions)

        kind = str(kind).lower()

        if kind == "arch":
            series, info = self.generate_arch_series(self.length)

        elif kind == "garch":
            series, info = self.generate_garch_series(self.length)

        elif kind == "egarch":
            series, info = self.generate_egarch_series(self.length)

        elif kind == "aparch":
            series, info = self.generate_aparch_series(self.length)

        else:
            raise ValueError(
                f"Invalid volatility process '{kind}'. "
                f"Choose from {self.volatile_base_distributions}."
            )

        # Used as external innovations for another base model.
        if as_innovations:
            return np.asarray(series, dtype=float), info

        # Standalone volatility base series.
        df = pd.DataFrame({
            "time": np.arange(self.length),
            "data": np.asarray(series, dtype=float),
            "stationary": np.ones(self.length,dtype=int),
            "seasonal": np.zeros(self.length,dtype=int)})

        return df, info

    def generate_stationary_base_series(
        self,
        distribution=None,
        innovations=None
    ):
        if distribution is None:
            distribution = np.random.choice(
                self.stationary_base_distributions
            )

        if distribution == 'white_noise':
            if innovations is not None:
                raise ValueError(
                    "white_noise cannot be combined with an external "
                    "innovation process."
                )

            series, info = self.generate_white_noise(
                self.length
            )

        elif distribution == 'ar':
            series, info = self.generate_ar_series(
                self.length,
                innovations=innovations
            )

        elif distribution == 'ma':
            series, info = self.generate_ma_series(
                self.length,
                innovations=innovations
            )

        elif distribution == 'arma':
            series, info = self.generate_arma_series(
                self.length,
                innovations=innovations
            )

        else:
            raise ValueError(
                "Invalid stationary distribution. "
                "Choose from 'ar', 'ma', 'arma', or 'white_noise'."
            )

        df = pd.DataFrame({
            'time': np.arange(self.length),
            'data': series,
            'stationary': np.ones(self.length, dtype=int),
            'seasonal': np.zeros(self.length, dtype=int),
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
        Reconstruct one deterministic Fourier seasonal context
        from an existing seasonal base series.

        Supported:
            - single_seasonality
            - multiple_seasonality
            - DETERMINISTIC_SARMA
            - DETERMINISTIC_SARIMA
            - SEASONAL_UNIT_ROOT_FOURIER

        Pure SARMA / SARIMA are intentionally unsupported because
        they do not contain a deterministic Fourier component.
        """

        # =====================================================
        # PERIOD
        # =====================================================

        periods = seasonal_info.get("periods")

        if periods is None or len(periods) == 0:
            raise ValueError(
                "seasonal_info must contain at least "
                "one seasonal period."
            )

        period = int(
            random.choice(periods)
        )

        subtype = str(
            seasonal_info.get("subtype", "")
        ).lower()

        # =====================================================
        # FOURIER COEFFICIENTS
        # =====================================================

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
                    if int(item["period"]) == period
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

            # In single/multiple generation, coefficients are
            # stored before the final scale_factor is applied.
            coefficient_scale = seasonal_info.get(
                "scale_factor",
                1.0
            )

        elif subtype == "single_seasonality":

            coefficients = seasonal_info.get(
                "coefficients"
            )

            if coefficients is None:
                raise ValueError(
                    "No Fourier coefficients found "
                    "for single seasonality."
                )

            coefficient_scale = seasonal_info.get(
                "scale_factor",
                1.0
            )

        elif subtype in {
            "deterministic_sarma",
            "deterministic_sarima",
            "seasonal_unit_root_fourier",
        }:

            coefficients = seasonal_info.get(
                "fourier_coefficients"
            )

            if coefficients is None:
                raise ValueError(
                    f"No Fourier coefficients found "
                    f"for subtype={subtype}."
                )

            # These generators already store the FINAL scaled
            # and strength-calibrated Fourier coefficients.
            coefficient_scale = 1.0

        elif subtype in {
            "pure_sarma",
            "pure_sarima",
        }:

            raise ValueError(
                f"Contextual anomaly generation requires an "
                f"explicit deterministic seasonal component. "
                f"{subtype} contains stochastic seasonality only."
            )

        else:

            raise ValueError(
                f"Unsupported seasonal subtype for contextual "
                f"anomaly generation: {subtype}"
            )

        # =====================================================
        # RECONSTRUCT FOURIER F_t
        # =====================================================

        t = np.arange(n)

        fourier_term = np.zeros(
            n,
            dtype=float
        )

        for coefficient in coefficients:

            k = int(
                coefficient["harmonic"]
            )

            sin_coef = float(
                coefficient["sin_coef"]
            )

            cos_coef = float(
                coefficient["cos_coef"]
            )

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

        fourier_term *= coefficient_scale

        # =====================================================
        # OBSERVED-DOMAIN SEASONAL CONTEXT
        # =====================================================

        if subtype == "seasonal_unit_root_fourier":

            # Special model:
            #
            #   (1-B^s)Y_t = F_t + u_t
            #
            # Fourier therefore lives in the seasonal-
            # difference equation. To obtain its contribution
            # in the observed Y_t domain, integrate it
            # seasonally.

            seasonal_context = np.zeros(
                n,
                dtype=float
            )

            for i in range(
                period,
                n
            ):
                seasonal_context[i] = (
                    seasonal_context[i - period]
                    + fourier_term[i]
                )

        else:

            # single / multiple /
            # deterministic SARMA /
            # deterministic SARIMA
            #
            # Fourier already exists directly in level domain:
            #
            #   Y_t = F_t + background_t

            seasonal_context = (
                fourier_term
            )

        return period, seasonal_context
    
    def generate_contextual_anomalies(
        self,
        df,
        seasonal_info,
        num_anomalies=1,
        location=None,
        anomaly_strength=1,
        max_attempts=10,
        is_loc=None,
        scale_factor=1
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
        if slope is None:
            slope = random.uniform(0.05,0.5) / (len(series) / 100)
        slope = sign * abs(slope)        
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
        if a is None:
            a = random.uniform(2.0, 5.0)

        a = sign * abs(a)
    
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
        trend = sign * (
            a * t**3
            + b * t**2
            + c * t
        ) * abs(amplitude)
    
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
        damping_rate = damping_rate if damping_rate is not None else random.uniform(0.005, 0.01)
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

    def generate_stochastic_trend(
        self,
        kind='rw',
        d=1,
        const=False,
        drift=None,
        noise_std=1.0,
        innovations=None
    ):
        t = np.arange(
            self.length
        )

        # -----------------------------------------------------
        # Innovation sequence
        # -----------------------------------------------------

        if innovations is None:

            noise = np.random.normal(
                0,
                noise_std,
                self.length
            )

        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != self.length:
                raise ValueError(
                    f"innovations length must match series length. "
                    f"Expected {self.length}, "
                    f"got {len(innovations)}."
                )

            noise = innovations

        # -----------------------------------------------------
        # Random Walk
        # -----------------------------------------------------

        if kind == 'rw':

            info = {
                'type': 'trend',
                'subtype': 'random_walk',
                'drift': None
            }

            series = np.cumsum(
                noise
            )

        # -----------------------------------------------------
        # Random Walk with Drift
        # -----------------------------------------------------

        elif kind == 'rwd':

            if drift is None:

                drift = np.random.uniform(
                    0.01,
                    0.1
                )

            info = {
                'type': 'trend',
                'subtype': 'random_walk_with_drift',
                'drift': drift
            }

            series = (
                drift * t
                + np.cumsum(noise)
            )

        # -----------------------------------------------------
        # ARI
        # -----------------------------------------------------

        elif kind == 'ari':

            series, info = (
                self.generate_ari_series(
                    length=self.length,
                    d=d,
                    const=const,
                    drift=drift,
                    innovations=innovations
                )
            )

        # -----------------------------------------------------
        # IMA
        # -----------------------------------------------------

        elif kind == 'ima':

            series, info = (
                self.generate_ima_series(
                    length=self.length,
                    d=d,
                    const=const,
                    drift=drift,
                    innovations=innovations
                )
            )

        # -----------------------------------------------------
        # ARIMA
        # -----------------------------------------------------

        elif kind == 'arima':

            series, info = (
                self.generate_arima_series(
                    length=self.length,
                    d=d,
                    const=const,
                    drift=drift,
                    innovations=innovations
                )
            )

        else:

            raise ValueError(
                "Invalid kind. Choose from "
                "'rw', 'rwd', 'ari', 'ima', or 'arima'."
            )

        df = pd.DataFrame({
            'time': np.arange(
                self.length
            ),
            'data': series,
            'stationary': np.zeros(
                self.length,
                dtype=int
            ),
            'seasonal': np.zeros(
                self.length,
                dtype=int
            ),
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

    def _build_fourier_component(
        self,
        periods,
        amplitudes,
        num_harmonics=1):
        """
        Build a deterministic Fourier seasonal component.

        This function ONLY builds the Fourier signal.

        It does NOT:
        - generate a background process
        - add Gaussian noise
        - calibrate against a background
        - create a DataFrame

        Parameters
        ----------
        periods : int or list[int]
            Seasonal periods.

        amplitudes : float or list[float]
            Base amplitude for each seasonal period.

        num_harmonics : int
            Number of Fourier harmonics per period.

        Returns
        -------
        fourier : np.ndarray
            Raw deterministic Fourier component.

        coefficients_by_period : list[dict]
            Fourier coefficients grouped by seasonal period.
        """

        periods = self.normalize_period_list(periods)

        if periods is None or len(periods) == 0:
            raise ValueError(
                "At least one seasonal period is required."
            )

        # Allow scalar amplitude for a single period.
        if np.isscalar(amplitudes):
            amplitudes = [float(amplitudes)]
        else:
            amplitudes = list(amplitudes)

        if len(amplitudes) != len(periods):
            raise ValueError("Length of amplitudes must match length of periods.")

        if num_harmonics < 1:
            raise ValueError("num_harmonics must be at least 1.")

        n = self.length
        t = np.arange(n)

        fourier = np.zeros(n,dtype=float)

        coefficients_by_period = []

        for period, amplitude in zip(periods,amplitudes):
            period = int(period)
            amplitude = float(amplitude)

            if period <= 0:
                raise ValueError("Seasonal periods must be positive.")

            period_coefficients = []

            for k in range(1,num_harmonics + 1):
                A_k = (amplitude * np.random.uniform(0.5, 1.0) / k)
                B_k = (amplitude * np.random.uniform(0.5, 1.0) / k)
                fourier += (A_k * np.sin(2 * np.pi * k * t / period))
                fourier += (B_k * np.cos(2 * np.pi * k * t / period))
                period_coefficients.append({
                    "harmonic": k,
                    "sin_coef": A_k,
                    "cos_coef": B_k})

            coefficients_by_period.append({
                "period": period,
                "coefficients":period_coefficients})

        return (fourier, coefficients_by_period)


    def _calibrate_fourier_to_background(
        self,
        fourier,
        background,
        difference_order=0,
        seasonal_strength_range=(0.8, 2.0)):

        """
        Scale a deterministic Fourier component relative to
        the background process.

        For stationary / volatility backgrounds:
            difference_order = 0

        For integrated stochastic backgrounds:
            difference_order = integration order d
        """

        fourier = np.asarray(fourier,dtype=float).copy()

        reference = np.asarray(background, dtype=float).copy()
        fourier_reference = np.asarray(fourier, dtype=float).copy()

        for _ in range(difference_order):
            reference = np.diff(reference)
            fourier_reference = np.diff(fourier_reference)

        background_scale = np.std(reference)
        fourier_scale = np.std(fourier_reference)

        seasonal_strength = (np.random.uniform(seasonal_strength_range[0],seasonal_strength_range[1]))

        if(background_scale <= 1e-8 or fourier_scale <= 1e-8):
            return (fourier, 1.0, seasonal_strength)

        calibration_factor = (seasonal_strength * background_scale / fourier_scale)

        fourier *= calibration_factor

        return (fourier,calibration_factor,seasonal_strength)

    def compose_with_fourier_seasonality(
        self,
        background_df,
        kind="single",
        period=None,
        periods=None,
        amplitude=None,
        amplitudes=None,
        num_components=2,
        num_harmonics=1,
        scale_factor=1.0,
        difference_order=0,
        seasonal_strength_range=(0.8, 2.0),
        allowed_periods=None,
        min_cycles=6
    ):
        """
        Add deterministic Fourier seasonality to an existing background.

            Y_t = B_t + F_t

        For multiple seasonality with an integrated background,
        each seasonal period is balanced separately in the
        differenced domain before global Fourier calibration.
        """

        if "data" not in background_df.columns:
            raise ValueError("background_df must contain a 'data' column.")

        if len(background_df) != self.length:
            raise ValueError("background_df length must match generator length.")

        kind = str(kind).lower()

        if kind not in {"single", "multiple"}:
            raise ValueError("kind must be 'single' or 'multiple'.")

        # =====================================================
        # PERIODS + AMPLITUDES
        # =====================================================

        if kind == "single":
            selected_period, _ = self.choose_calendar_period(
                period=period,
                allowed_periods=allowed_periods,
                min_cycles=min_cycles
            )

            selected_periods = [selected_period]
            selected_amplitudes = [1.0 if amplitude is None else float(amplitude)]

        else:
            if allowed_periods is None:
                allowed_periods = self.get_all_calendar_periods()

            allowed_periods = sorted(set(int(p) for p in allowed_periods))

            valid_periods = self.get_valid_calendar_periods(
                allowed_periods=allowed_periods,
                min_cycles=min_cycles
            )

            selected_periods = self.normalize_period_list(periods)

            if selected_periods is None:
                if len(valid_periods) < num_components:
                    raise ValueError(
                        f"Multiple seasonality requires {num_components} valid periods, "
                        f"but only {valid_periods} are available."
                    )

                selected_periods = random.sample(valid_periods, num_components)

            else:
                if len(selected_periods) < 2:
                    raise ValueError("Multiple seasonality requires at least two periods.")

                for p in selected_periods:
                    if p not in allowed_periods:
                        raise ValueError(f"period={p} is not in allowed calendar periods.")

                    if p not in valid_periods:
                        raise ValueError(
                            f"period={p} is too large for length={self.length}."
                        )

            if amplitudes is None:
                selected_amplitudes = [1.0] * len(selected_periods)

            elif np.isscalar(amplitudes):
                selected_amplitudes = [float(amplitudes)] * len(selected_periods)

            else:
                selected_amplitudes = list(amplitudes)

                if len(selected_amplitudes) != len(selected_periods):
                    raise ValueError(
                        "Length of amplitudes must match length of periods."
                    )

        # =====================================================
        # BUILD FOURIER
        # =====================================================

        period_balance_factors = {}

        # -----------------------------------------------------
        # MULTIPLE + INTEGRATED BACKGROUND
        #
        # Build each period separately and compensate for the
        # attenuation caused by differencing.
        # -----------------------------------------------------

        if kind == "multiple" and difference_order > 0:
            fourier = np.zeros(self.length, dtype=float)
            coefficients_by_period = []

            for p, amp in zip(selected_periods, selected_amplitudes):
                component, component_meta = self._build_fourier_component(
                    periods=[p],
                    amplitudes=[amp],
                    num_harmonics=num_harmonics
                )

                diff_component = component.copy()

                for _ in range(difference_order):
                    diff_component = np.diff(diff_component)

                level_std = np.std(component)
                diff_std = np.std(diff_component)

                if level_std > 1e-8 and diff_std > 1e-8:
                    balance_factor = level_std / diff_std
                else:
                    balance_factor = 1.0

                component *= balance_factor

                # Bake the period-specific balance into its coefficients.
                for coef in component_meta[0]["coefficients"]:
                    coef["sin_coef"] *= balance_factor
                    coef["cos_coef"] *= balance_factor

                fourier += component
                coefficients_by_period.append(component_meta[0])
                period_balance_factors[int(p)] = float(balance_factor)

        # -----------------------------------------------------
        # SINGLE OR NON-INTEGRATED MULTIPLE
        # -----------------------------------------------------

        else:
            fourier, coefficients_by_period = self._build_fourier_component(
                periods=selected_periods,
                amplitudes=selected_amplitudes,
                num_harmonics=num_harmonics
            )

            period_balance_factors = {
                int(p): 1.0
                for p in selected_periods
            }

        # =====================================================
        # USER SCALE
        # =====================================================

        fourier *= scale_factor

        # =====================================================
        # GLOBAL CALIBRATION AGAINST BACKGROUND
        # =====================================================

        fourier, calibration_factor, seasonal_strength = (
            self._calibrate_fourier_to_background(
                fourier=fourier,
                background=background_df["data"].to_numpy(dtype=float),
                difference_order=difference_order,
                seasonal_strength_range=seasonal_strength_range
            )
        )

        final_scale_factor = scale_factor * calibration_factor

        # =====================================================
        # COMBINE
        # =====================================================

        df = background_df.copy()
        df.loc[:, "data"] = df["data"].to_numpy(dtype=float) + fourier
        df.loc[:, "stationary"] = 0
        df.loc[:, "seasonal"] = 1

        if kind == "single":
            df.loc[:, "single_seas"] = 1
        else:
            df.loc[:, "multiple_seas"] = 1

        # =====================================================
        # METADATA
        # =====================================================

        period_meanings = {
            p: self.get_period_meanings(p)
            for p in selected_periods
        }

        if kind == "single":
            info = {
                "type": "seasonal",
                "subtype": "single_seasonality",
                "periods": selected_periods,
                "period_meanings": period_meanings,
                "amplitudes": selected_amplitudes[0],
                "num_harmonics": num_harmonics,
                "coefficients": coefficients_by_period[0]["coefficients"],
                "scale_factor": final_scale_factor,
                "seasonal_strength": seasonal_strength,
                "period_balance_factors": period_balance_factors,
                "composition_mode": True,
                "calibration_difference_order": int(difference_order),
            }

        else:
            info = {
                "type": "seasonal",
                "subtype": "multiple_seasonality",
                "periods": selected_periods,
                "period_meanings": period_meanings,
                "amplitudes": selected_amplitudes,
                "num_harmonics": num_harmonics,
                "coefficients": coefficients_by_period,
                "scale_factor": final_scale_factor,
                "seasonal_strength": seasonal_strength,
                "period_balance_factors": period_balance_factors,
                "composition_mode": True,
                "calibration_difference_order": int(difference_order),
            }

        return df, info

# SEASONALITY

    def generate_single_seasonality(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1,
        num_harmonics=1,
        allowed_periods=None,
        min_cycles=6):
        n = self.length

        # STANDALONE BACKGROUND

        series = np.random.normal(loc=0.0,scale=0.2,size=n)

        # SEASONAL NOISE

        if noise_std is None:
            noise_std = np.random.uniform(0.01,0.05)

        # PERIOD

        period, _ = (
            self.choose_calendar_period(
                period=period,
                allowed_periods=allowed_periods,
                min_cycles=min_cycles))

        # AMPLITUDE

        if amplitude is None:
            base_std = np.std(series)

            amplitude = (base_std * np.random.uniform(0.5,2.5))

        # FOURIER

        (fourier,coefficients_by_period) = self._build_fourier_component(
            periods=[period],
            amplitudes=[amplitude],
            num_harmonics=num_harmonics)

        coefficients = (coefficients_by_period[0]["coefficients"])

        # STANDALONE SEASONAL NOISE

        seasonal_noise = (np.random.normal(0,noise_std,size=n))

        seasonality = (fourier + seasonal_noise)

        series += (seasonality * scale_factor)

        # DATAFRAME

        df = pd.DataFrame({
            "time":np.arange(n),
            "data":series,
            "stationary":np.zeros(n,dtype=int),
            "seasonal":np.ones(n,dtype=int),
            "single_seas":np.ones(n,dtype=int)})

        # METADATA

        info = {
            "type":"seasonal",
            "subtype":"single_seasonality",
            "periods":[period],
            "period_meanings": {period:self.get_period_meanings(period)},
            "amplitudes":amplitude,
            "noise_std":noise_std,
            "scale_factor":scale_factor,
            "num_harmonics":num_harmonics,
            "coefficients":coefficients}

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
        min_cycles=6):
        n = self.length

        # STANDALONE BACKGROUND

        series = np.random.normal(loc=0.0,scale=0.2,size=n)

        # SEASONAL NOISE

        if noise_std is None:
            noise_std = np.random.uniform(0.01,0.05)

        # VALID PERIODS
        # 

        if allowed_periods is None:
            allowed_periods = (self.get_all_calendar_periods())

        allowed_periods = sorted(set(int(p) for p in allowed_periods))

        valid_periods = (
            self.get_valid_calendar_periods(allowed_periods=allowed_periods, min_cycles=min_cycles))

        periods = (self.normalize_period_list(periods))

        # PERIOD SELECTION

        if periods is None:
            if (len(valid_periods) < num_components):
                raise ValueError(f"Multiple seasonality needs {num_components} valid periods, but only {valid_periods} are available.")

            periods = random.sample(valid_periods,num_components)

        else:
            if len(periods) < 2:
                raise ValueError("Multiple seasonality requires at least 2 periods.")

            for p in periods:
                if p not in allowed_periods:
                    raise ValueError(f"period={p} is not in allowed calendar periods.")

                if p not in valid_periods:
                    raise ValueError(f"period={p} is too large for length={n}.")

        # AMPLITUDES

        if amplitudes is None:
            base_std = np.std(series)

            amplitudes = [(base_std * np.random.uniform(0.5,2.0)) for _ in periods]

        else:
            if np.isscalar(amplitudes):
                amplitudes = [float(amplitudes) for _ in periods]

            else:
                amplitudes = list(amplitudes)

            if (len(amplitudes) != len(periods)):
                raise ValueError("Length of amplitudes must match length of periods.")

        # FOURIER

        (fourier,coefficients_meta) = self._build_fourier_component(
            periods=periods,
            amplitudes=amplitudes,
            num_harmonics=num_harmonics)

        # STANDALONE SEASONAL NOISE
        seasonal_noise = (np.random.normal(0,noise_std,size=n))

        seasonality = (fourier + seasonal_noise)

        series += (seasonality * scale_factor)

        # DATAFRAME

        df = pd.DataFrame({
            "time":np.arange(n),
            "data":series,
            "stationary":np.zeros(n,dtype=int),
            "seasonal":np.ones(n,dtype=int),
            "multiple_seas":np.ones(n,dtype=int)})

        # METADATA

        info = {
            "type": "seasonal",
            "subtype": "multiple_seasonality",
            "periods": periods,
            "period_meanings": {p: self.get_period_meanings(p) for p in periods},
            "amplitudes": amplitudes,
            "noise_std": noise_std,
            "scale_factor": scale_factor,
            "num_harmonics": num_harmonics,
            "coefficients": coefficients_meta}

        return df, info

    def generate_pure_sarma(
        self,
        period=None,
        noise_std=None,
        order_range=(1, 2),
        seasonal_coef_range=(-0.4, 0.4),
        min_seasonal_abs=0.20,
        min_cancellation_gap=0.10,
        allowed_periods=None,
        min_cycles=6,
        max_attempts=1000
    ):
        """
        Pure multiplicative stochastic SARMA process.

        Model
        -----
            phi(B) Phi(B^s) Y_t
                =
            theta(B) Theta(B^s) epsilon_t

        Properties
        ----------
        - d = 0
        - D = 0
        - no deterministic Fourier component
        - no extra observation noise
        - seasonality comes entirely from seasonal AR/MA dynamics
        """

        n = self.length

        # SEASONAL PERIOD

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles)

        s = period

        if n <= s:
            raise ValueError("Series length must be larger than the seasonal period.")

        if noise_std is None:
            noise_std = np.random.uniform(0.1, 0.20)

        # HELPER: SEASONAL COEFFICIENT AWAY FROM ZERO
        def draw_seasonal_coef():
            for _ in range(1000):
                coef = np.random.uniform(
                    seasonal_coef_range[0],
                    seasonal_coef_range[1])
                if abs(coef) >= min_seasonal_abs:
                    return coef

            raise RuntimeError("Could not draw a valid seasonal coefficient.")

        arma_process = None

        # =====================================================
        # PARAMETER SEARCH
        # =====================================================

        for _ in range(max_attempts):

            # -------------------------------------------------
            # NON-SEASONAL ORDERS
            # -------------------------------------------------

            ar_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            ma_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            # -------------------------------------------------
            # SEASONAL ORDERS
            #
            # P,Q ∈ {0,1}
            # but not both zero
            # -------------------------------------------------

            while True:

                seasonal_ar_order = np.random.randint(0, 2)
                seasonal_ma_order = np.random.randint(0, 2)

                if (
                    seasonal_ar_order > 0
                    or seasonal_ma_order > 0
                ):
                    break

            # -------------------------------------------------
            # NON-SEASONAL COEFFICIENTS
            # -------------------------------------------------

            ar_coefs = np.random.uniform(
                -0.4,
                0.4,
                ar_order
            )

            ma_coefs = np.random.uniform(
                -0.4,
                0.4,
                ma_order
            )

            # -------------------------------------------------
            # SEASONAL COEFFICIENTS
            # -------------------------------------------------

            if seasonal_ar_order == 1:
                seasonal_ar_coefs = np.array([
                    draw_seasonal_coef()
                ])
            else:
                seasonal_ar_coefs = np.array([], dtype=float)

            if seasonal_ma_order == 1:
                seasonal_ma_coefs = np.array([
                    draw_seasonal_coef()
                ])
            else:
                seasonal_ma_coefs = np.array([], dtype=float)

            # -------------------------------------------------
            # PREVENT NEAR CANCELLATION
            #
            # Theta ≈ -Phi
            # -------------------------------------------------

            if (
                seasonal_ar_order == 1
                and seasonal_ma_order == 1
            ):

                phi = seasonal_ar_coefs[0]
                theta = seasonal_ma_coefs[0]

                seasonal_cancellation_gap = abs(
                    phi + theta
                )

                if (
                    seasonal_cancellation_gap
                    < min_cancellation_gap
                ):
                    continue

            else:
                seasonal_cancellation_gap = np.nan

            # =================================================
            # POLYNOMIALS
            # =================================================

            nonseasonal_ar = np.r_[
                1.0,
                -ar_coefs
            ]

            nonseasonal_ma = np.r_[
                1.0,
                ma_coefs
            ]

            seasonal_ar = np.zeros(
                seasonal_ar_order * s + 1
            )

            seasonal_ar[0] = 1.0

            if seasonal_ar_order == 1:
                seasonal_ar[s] = (
                    -seasonal_ar_coefs[0]
                )

            seasonal_ma = np.zeros(
                seasonal_ma_order * s + 1
            )

            seasonal_ma[0] = 1.0

            if seasonal_ma_order == 1:
                seasonal_ma[s] = (
                    seasonal_ma_coefs[0]
                )

            # =================================================
            # MULTIPLICATIVE SARMA
            # =================================================

            ar_poly = np.convolve(
                nonseasonal_ar,
                seasonal_ar
            )

            ma_poly = np.convolve(
                nonseasonal_ma,
                seasonal_ma
            )

            candidate = ArmaProcess(
                ar_poly,
                ma_poly
            )

            if (
                candidate.isstationary
                and candidate.isinvertible
            ):
                arma_process = candidate
                break

        if arma_process is None:
            raise RuntimeError(
                "Could not generate a valid pure SARMA process."
            )

        # =====================================================
        # GENERATE
        # =====================================================

        burnin = max(
            200,
            8 * s
        )

        series = arma_process.generate_sample(
            nsample=n,
            burnin=burnin,
            scale=noise_std
        )

        # =====================================================
        # DATAFRAME
        # =====================================================

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "stochastic_component": series.copy(),
            "stationary": np.ones(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "sarma": np.ones(n).astype(int)
        })

        # =====================================================
        # METADATA
        # =====================================================

        info = {
            "type": "seasonal",
            "subtype": "PURE_SARMA",

            "periods": [s],

            "period_meanings": {
                s: self.get_period_meanings(s)
            },

            "seasonality_source":
                "stochastic_seasonal_ar_ma",

            "diff": 0,
            "seasonal_diff": 0,

            "unit_root": "none",
            "seasonal_unit_root": "none",

            "noise_std": noise_std,

            "ar_order": ar_order,
            "ma_order": ma_order,

            "ar_coefs": ar_coefs,
            "ma_coefs": ma_coefs,

            "seasonal_ar_order": seasonal_ar_order,
            "seasonal_ma_order": seasonal_ma_order,

            "seasonal_ar_coefs": seasonal_ar_coefs,
            "seasonal_ma_coefs": seasonal_ma_coefs,

            "min_seasonal_abs": min_seasonal_abs,

            "seasonal_cancellation_gap":
                seasonal_cancellation_gap,

            "fourier_used": False
        }

        return df, info

    def generate_pure_sarima(
        self,
        period=None,
        noise_std=None,
        initial_std=0.2,
        d=0,
        D=1,
        order_range=(0, 2),
        seasonal_order_range=(0, 1),
        coef_range=(-0.4, 0.4),
        seasonal_coef_range=(-0.4, 0.4),
        allowed_periods=None,
        min_cycles=6,
        max_attempts=1000):
        """
        Pure multiplicative stochastic SARIMA process.

        Model
        -----
            phi(B) Phi(B^s)
            (1-B)^d (1-B^s)^D Y_t

                =

            theta(B) Theta(B^s) epsilon_t

        No deterministic Fourier component is used.
        """

        n = self.length

        # DIFFERENCING ORDERS

        if d not in (0, 1):
            raise ValueError("Currently d must be 0 or 1.")

        if D not in (0, 1):
            raise ValueError("Currently D must be 0 or 1.")

        # PERIOD

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles)

        s = period

        if n <= s:
            raise ValueError("Series length must be larger than seasonal period.")



        if noise_std is None:
            noise_std = np.random.uniform(0.1,0.20)

        arma_process = None

        # STATIONARY SARMA CORE

        for _ in range(max_attempts):
            ar_order = np.random.randint(order_range[0],order_range[1] + 1)

            ma_order = np.random.randint(order_range[0],order_range[1] + 1)

            seasonal_ar_order = np.random.randint(seasonal_order_range[0],seasonal_order_range[1] + 1)

            seasonal_ma_order = np.random.randint(seasonal_order_range[0],seasonal_order_range[1] + 1)

            # If D=0, ensure some stochastic seasonal mechanism exists.
            if (D == 0 and seasonal_ar_order == 0 and seasonal_ma_order == 0):
                continue

            # COEFFICIENTS

            if ar_order > 0:
                ar_coefs = np.random.uniform(coef_range[0],coef_range[1],ar_order)
            else:
                ar_coefs = np.array([], dtype=float)

            if ma_order > 0:
                ma_coefs = np.random.uniform(coef_range[0],coef_range[1],ma_order)
            else:
                ma_coefs = np.array([], dtype=float)

            if seasonal_ar_order > 0:
                seasonal_ar_coefs = np.random.uniform(seasonal_coef_range[0],seasonal_coef_range[1],seasonal_ar_order)
            else:
                seasonal_ar_coefs = np.array([], dtype=float)

            if seasonal_ma_order > 0:
                seasonal_ma_coefs = np.random.uniform(seasonal_coef_range[0],seasonal_coef_range[1],seasonal_ma_order)
            else:
                seasonal_ma_coefs = np.array([], dtype=float)

            # POLYNOMIALS

            nonseasonal_ar = np.r_[1.0,-ar_coefs]

            nonseasonal_ma = np.r_[1.0,ma_coefs]

            seasonal_ar = np.zeros(seasonal_ar_order * s + 1)

            seasonal_ar[0] = 1.0

            for i, coef in enumerate(seasonal_ar_coefs,start=1):
                seasonal_ar[i * s] = -coef

            seasonal_ma = np.zeros(seasonal_ma_order * s + 1)

            seasonal_ma[0] = 1.0

            for i, coef in enumerate(seasonal_ma_coefs,start=1):
                seasonal_ma[i * s] = coef

            ar_poly = np.convolve(nonseasonal_ar,seasonal_ar)

            ma_poly = np.convolve(nonseasonal_ma,seasonal_ma)

            candidate = ArmaProcess(ar_poly,ma_poly)

            if (candidate.isstationary and candidate.isinvertible):
                arma_process = candidate
                break

        if arma_process is None:
            raise RuntimeError("Could not generate valid SARMA core.")

        # STATIONARY SARMA CORE

        burnin = max(200,8 * s)

        sarma_core = arma_process.generate_sample(
            nsample=n,
            burnin=burnin,
            scale=noise_std)

        stochastic_component = sarma_core.copy()

        # SEASONAL INTEGRATION
        # (1-B^s)^D

        for _ in range(D):
            integrated = np.zeros(n)
            integrated[:s] = np.random.normal(0,initial_std,size=s)

            for i in range(s, n):
                integrated[i] = (integrated[i - s] + stochastic_component[i])

            stochastic_component = integrated

        # NON-SEASONAL INTEGRATION
        # (1-B)^d


        for _ in range(d):
            integrated = np.zeros(n)
            integrated[0] = np.random.normal(0,initial_std)

            for i in range(1, n):
                integrated[i] = (integrated[i - 1] + stochastic_component[i])

            stochastic_component = integrated

        series = stochastic_component

        # DATAFRAME

        df = pd.DataFrame({
            "time": np.arange(n),
            "data":series,
            "stationary_core": sarma_core,
            "stochastic_component": stochastic_component,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "sarima": np.ones(n).astype(int)})


        # METADATA

        info = {
            "type": "seasonal",
            "subtype": "PURE_SARIMA",
            "periods": [s],
            "period_meanings": {s: self.get_period_meanings(s)},
            "seasonality_source": "stochastic_sarima",
            "diff": d,
            "seasonal_diff": D,
            "unit_root":("unit_root" if d > 0 else "none"),
            "seasonal_unit_root":("seasonal_unit_root" if D > 0 else "none"),
            "noise_std": noise_std,
            "initial_std": initial_std,
            "ar_order": ar_order,
            "ma_order": ma_order,
            "ar_coefs": ar_coefs,
            "ma_coefs": ma_coefs,
            "seasonal_ar_order":seasonal_ar_order,
            "seasonal_ma_order":seasonal_ma_order,
            "seasonal_ar_coefs":seasonal_ar_coefs,
            "seasonal_ma_coefs":seasonal_ma_coefs,
            "fourier_used": False}

        return df, info

    def generate_deterministic_sarma(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1.0,
        num_harmonics=1,
        order_range=(1, 2),
        coef_range=(-0.4, 0.4),
        seasonal_strength_range=(0.8, 2.0),
        allowed_periods=None,
        min_cycles=6,
        max_attempts=1000,
        innovations=None,
        additional_component=None):
        """
        Project-specific deterministic seasonal ARMA model.

        IMPORTANT:
        This is NOT textbook SARMA.

        Standalone model
        ----------------
            Y_t = F_t + U_t

        where:

            phi(B) U_t
                =
            theta(B) epsilon_t

        and F_t is deterministic Fourier seasonality.

        Therefore the ONLY seasonal component is F_t.

        Composition support
        -------------------
        innovations:
            Optional external innovation sequence.

            Example:
                GARCH innovations -> ARMA -> Fourier

            This allows combinations such as deterministic SARMA
            with volatility-driven innovations.

        additional_component:
            Optional additional non-seasonal component added to
            the internally generated ARMA background.

            Example:
                ARMA_internal + AR_external + Fourier

            This allows technically composable base-family
            combinations without changing the seasonal definition.

        In standalone mode:

            additional_component = None
            innovations = None

        and the model reduces to:

            Y_t = U_t + F_t
        """

        n = self.length

        # PERIOD

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles)

        s = period

        # =====================================================
        # INNOVATION SCALE
        # =====================================================

        if noise_std is None:
            noise_std = np.random.uniform(
                0.1,
                0.20
            )

        # =====================================================
        # NON-SEASONAL ARMA PARAMETER GENERATION
        # =====================================================

        arma_process = None

        for _ in range(max_attempts):

            ar_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            ma_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            ar_coefs = np.random.uniform(
                coef_range[0],
                coef_range[1],
                ar_order
            )

            ma_coefs = np.random.uniform(
                coef_range[0],
                coef_range[1],
                ma_order
            )

            ar_poly = np.r_[
                1.0,
                -ar_coefs
            ]

            ma_poly = np.r_[
                1.0,
                ma_coefs
            ]

            candidate = ArmaProcess(
                ar_poly,
                ma_poly
            )

            if (
                candidate.isstationary
                and candidate.isinvertible
            ):
                arma_process = candidate
                break

        if arma_process is None:
            raise RuntimeError(
                "Could not generate valid ARMA background."
            )

        # =====================================================
        # GENERATE NON-SEASONAL ARMA BACKGROUND
        # =====================================================

        burnin = 200

        # -----------------------------------------------------
        # Standard standalone SARMA:
        # Gaussian innovations generated internally.
        # -----------------------------------------------------

        if innovations is None:

            stochastic_component = (
                arma_process.generate_sample(
                    nsample=n,
                    burnin=burnin,
                    scale=noise_std
                )
            )

        # -----------------------------------------------------
        # External innovation process:
        # e.g. GARCH -> ARMA
        # -----------------------------------------------------

        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != n:
                raise ValueError(
                    "innovations length must match "
                    f"series length. Expected {n}, "
                    f"got {len(innovations)}."
                )

            # External innovations already contain their own
            # scale/dynamics, so do NOT apply noise_std again.
            #
            # We also do not use burnin here because the supplied
            # innovation sequence contains exactly n observations.
            stochastic_component = (
                arma_process.generate_sample(
                    nsample=n,
                    scale=1.0,
                    distrvs=lambda size: innovations
                )
            )

        # =====================================================
        # OPTIONAL ADDITIONAL NON-SEASONAL COMPONENT
        # =====================================================

        nonseasonal_background = (
            stochastic_component.copy()
        )

        if additional_component is not None:

            # Accept either:
            #   - a DataFrame containing "data"
            #   - ndarray / list / other array-like object

            if isinstance(
                additional_component,
                pd.DataFrame
            ):

                if (
                    "data"
                    not in additional_component.columns
                ):
                    raise ValueError(
                        "additional_component DataFrame "
                        "must contain a 'data' column."
                    )

                external_component = (
                    additional_component[
                        "data"
                    ].to_numpy(
                        dtype=float
                    )
                )

            else:

                external_component = np.asarray(
                    additional_component,
                    dtype=float
                )

            if len(external_component) != n:
                raise ValueError(
                    "additional_component length must "
                    f"match series length. Expected {n}, "
                    f"got {len(external_component)}."
                )

            nonseasonal_background = (
                nonseasonal_background
                + external_component
            )

        # =====================================================
        # FOURIER SEASONALITY
        # =====================================================

        if amplitude is None:
            amplitude = 1.0

        (
            fourier,
            coefficients_by_period
        ) = self._build_fourier_component(
            periods=[s],
            amplitudes=[amplitude],
            num_harmonics=num_harmonics
        )

        # Raw Fourier coefficients returned by the shared
        # Fourier builder.
        fourier_coefficients = (
            coefficients_by_period[0][
                "coefficients"
            ]
        )

        # -----------------------------------------------------
        # Initial user/config supplied Fourier scaling
        # -----------------------------------------------------

        fourier *= scale_factor

        # =====================================================
        # CALIBRATE FOURIER AGAINST FINAL NON-SEASONAL
        # BACKGROUND
        # =====================================================

        (
            fourier,
            calibration_factor,
            seasonal_strength
        ) = self._calibrate_fourier_to_background(
            fourier=fourier,
            background=nonseasonal_background,
            difference_order=0,
            seasonal_strength_range=
                seasonal_strength_range
        )

        # =====================================================
        # UPDATE FOURIER COEFFICIENTS TO FINAL SCALE
        # =====================================================

        # _build_fourier_component returned RAW coefficients.
        #
        # The Fourier signal experienced:
        #
        #   raw
        #     * scale_factor
        #     * calibration_factor
        #
        # Metadata must contain the FINAL coefficients because
        # _get_fourier_context() reconstructs deterministic SARMA
        # Fourier directly from these stored coefficients.

        total_fourier_scale = (
            scale_factor
            * calibration_factor
        )

        for coef_info in fourier_coefficients:

            coef_info["sin_coef"] *= (
                total_fourier_scale
            )

            coef_info["cos_coef"] *= (
                total_fourier_scale
            )

        # =====================================================
        # FINAL SERIES
        # =====================================================

        series = (
            nonseasonal_background
            + fourier
        )

        # =====================================================
        # DATAFRAME
        # =====================================================

        df = pd.DataFrame({
            "time":
                np.arange(n),

            "data":
                series,

            "stationary":
                np.zeros(
                    n,
                    dtype=int
                ),

            "seasonal":
                np.ones(
                    n,
                    dtype=int
                ),

            "sarma":
                np.ones(
                    n,
                    dtype=int
                )
        })

        # =====================================================
        # METADATA
        # =====================================================

        info = {
            "type":
                "seasonal",

            "subtype":
                "DETERMINISTIC_SARMA",

            "periods":
                [s],

            "period_meanings": {
                s:
                    self.get_period_meanings(
                        s
                    )
            },

            # Deterministic SARMA has no integration.
            "diff":
                0,

            "seasonal_diff":
                0,

            "unit_root":
                "none",

            "seasonal_unit_root":
                "none",

            "noise_std":
                noise_std,

            # Non-seasonal ARMA parameters
            "ar_order":
                ar_order,

            "ma_order":
                ma_order,

            "ar_coefs":
                ar_coefs,

            "ma_coefs":
                ma_coefs,

            # No stochastic seasonal AR/MA terms.
            "seasonal_ar_order":
                0,

            "seasonal_ma_order":
                0,

            "seasonal_ar_coefs":
                np.array(
                    [],
                    dtype=float
                ),

            "seasonal_ma_coefs":
                np.array(
                    [],
                    dtype=float
                ),

            # Deterministic Fourier metadata
            "num_harmonics":
                num_harmonics,

            "fourier_coefficients":
                fourier_coefficients,

            "fourier_used":
                True,

            # Composition / calibration metadata
            "seasonal_strength":
                seasonal_strength,

            "fourier_scale_factor":
                total_fourier_scale,

            "external_innovations_used":
                innovations is not None,

            "additional_component_used":
                additional_component is not None
        }

        return df, info

    def generate_deterministic_sarima(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1.0,
        initial_std=0.2,
        num_harmonics=1,
        d=1,
        order_range=(0, 2),
        coef_range=(-0.4, 0.4),
        seasonal_strength_range=(0.8, 2.0),
        allowed_periods=None,
        min_cycles=6,
        max_attempts=1000,
        innovations=None,
        additional_component=None
    ):
        """
        Project-specific deterministic seasonal ARIMA model.

        IMPORTANT:
        This is NOT textbook SARIMA.

        Standalone model
        ----------------
            Y_t = F_t + Z_t

        where:

            phi(B) (1-B)^d Z_t
                =
            theta(B) epsilon_t

        and F_t is deterministic Fourier seasonality.

        There is deliberately:

            - NO seasonal AR
            - NO seasonal MA
            - NO seasonal differencing D

        Therefore Fourier is the ONLY source of seasonality.

        Composition support
        -------------------
        innovations:
            Optional external innovation sequence.

            Example:
                GARCH innovations -> ARIMA -> Fourier

        additional_component:
            Optional additional non-seasonal component added
            AFTER ARIMA integration.

            Example:
                ARIMA_internal + external_component + Fourier
        """

        n = self.length

        # =====================================================
        # DIFFERENCING ORDER
        # =====================================================

        if d not in (0, 1):
            raise ValueError(
                "Currently d must be 0 or 1."
            )

        # =====================================================
        # PERIOD
        # =====================================================

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        s = period

        # =====================================================
        # INNOVATION SCALE
        # =====================================================

        if noise_std is None:
            noise_std = np.random.uniform(
                0.1,
                0.20
            )

        # =====================================================
        # NON-SEASONAL ARMA CORE PARAMETER GENERATION
        # =====================================================

        arma_process = None

        for _ in range(max_attempts):

            ar_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            ma_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            if ar_order > 0:
                ar_coefs = np.random.uniform(
                    coef_range[0],
                    coef_range[1],
                    ar_order
                )
            else:
                ar_coefs = np.array([], dtype=float)

            if ma_order > 0:
                ma_coefs = np.random.uniform(
                    coef_range[0],
                    coef_range[1],
                    ma_order
                )
            else:
                ma_coefs = np.array([], dtype=float)

            ar_poly = np.r_[1.0, -ar_coefs]
            ma_poly = np.r_[1.0, ma_coefs]

            candidate = ArmaProcess(
                ar_poly,
                ma_poly
            )

            if (
                candidate.isstationary
                and candidate.isinvertible
            ):
                arma_process = candidate
                break

        if arma_process is None:
            raise RuntimeError(
                "Could not generate valid ARMA core."
            )

        # =====================================================
        # GENERATE STATIONARY ARMA CORE
        # =====================================================

        burnin = 200

        if innovations is None:

            arma_core = (
                arma_process.generate_sample(
                    nsample=n,
                    burnin=burnin,
                    scale=noise_std
                )
            )

        else:

            innovations = np.asarray(
                innovations,
                dtype=float
            )

            if len(innovations) != n:
                raise ValueError(
                    "innovations length must match "
                    f"series length. Expected {n}, "
                    f"got {len(innovations)}."
                )

            arma_core = (
                arma_process.generate_sample(
                    nsample=n,
                    scale=1.0,
                    distrvs=lambda size: innovations
                )
            )

        # =====================================================
        # INTEGRATE d TIMES
        # =====================================================

        stochastic_component = arma_core.copy()

        for _ in range(d):

            integrated = np.zeros(n)

            integrated[0] = np.random.normal(
                0,
                initial_std
            )

            for i in range(1, n):
                integrated[i] = (
                    integrated[i - 1]
                    + stochastic_component[i]
                )

            stochastic_component = integrated

        # =====================================================
        # OPTIONAL ADDITIONAL NON-SEASONAL COMPONENT
        # =====================================================

        nonseasonal_background = (
            stochastic_component.copy()
        )

        if additional_component is not None:

            if isinstance(
                additional_component,
                pd.DataFrame
            ):

                if (
                    "data"
                    not in additional_component.columns
                ):
                    raise ValueError(
                        "additional_component DataFrame "
                        "must contain a 'data' column."
                    )

                external_component = (
                    additional_component[
                        "data"
                    ].to_numpy(dtype=float)
                )

            else:

                external_component = np.asarray(
                    additional_component,
                    dtype=float
                )

            if len(external_component) != n:
                raise ValueError(
                    "additional_component length must "
                    f"match series length. Expected {n}, "
                    f"got {len(external_component)}."
                )

            nonseasonal_background = (
                nonseasonal_background
                + external_component
            )

        # =====================================================
        # FOURIER SEASONALITY
        # =====================================================

        if amplitude is None:
            amplitude = 1.0

        (
            fourier,
            coefficients_by_period
        ) = self._build_fourier_component(
            periods=[s],
            amplitudes=[amplitude],
            num_harmonics=num_harmonics
        )

        fourier_coefficients = (
            coefficients_by_period[0][
                "coefficients"
            ]
        )

        # Initial config/user scale
        fourier *= scale_factor

        # =====================================================
        # CALIBRATE FOURIER AGAINST FINAL NON-SEASONAL
        # BACKGROUND
        # =====================================================

        (
            fourier,
            calibration_factor,
            seasonal_strength
        ) = self._calibrate_fourier_to_background(
            fourier=fourier,
            background=nonseasonal_background,
            difference_order=d,
            seasonal_strength_range=seasonal_strength_range
        )

        # =====================================================
        # UPDATE FOURIER COEFFICIENTS TO FINAL SCALE
        # =====================================================

        total_fourier_scale = (
            scale_factor
            * calibration_factor
        )

        for coef_info in fourier_coefficients:
            coef_info["sin_coef"] *= (
                total_fourier_scale
            )
            coef_info["cos_coef"] *= (
                total_fourier_scale
            )

        # =====================================================
        # FINAL SERIES
        # =====================================================

        series = (
            nonseasonal_background
            + fourier
        )

        # =====================================================
        # DATAFRAME
        # =====================================================

        df = pd.DataFrame({
            "time": np.arange(n),
            "data": series,
            "arma_core": arma_core,
            "stochastic_component": stochastic_component,
            "fourier_component": fourier,
            "stationary": np.zeros(n).astype(int),
            "seasonal": np.ones(n).astype(int),
            "sarima": np.ones(n).astype(int)
        })

        # =====================================================
        # METADATA
        # =====================================================

        info = {
            "type": "seasonal",
            "subtype": "DETERMINISTIC_SARIMA",
            "periods": [s],
            "period_meanings": {
                s: self.get_period_meanings(s)
            },

            "diff": d,
            "seasonal_diff": 0,

            "unit_root": (
                "unit_root"
                if d > 0 else "none"
            ),

            "seasonal_unit_root": "none",

            "noise_std": noise_std,
            "initial_std": initial_std,

            "ar_order": ar_order,
            "ma_order": ma_order,
            "ar_coefs": ar_coefs,
            "ma_coefs": ma_coefs,

            "seasonal_ar_order": 0,
            "seasonal_ma_order": 0,

            "seasonal_ar_coefs": np.array([], dtype=float),
            "seasonal_ma_coefs": np.array([], dtype=float),

            "num_harmonics": num_harmonics,
            "fourier_coefficients": fourier_coefficients,
            "fourier_used": True,

            "seasonal_strength": seasonal_strength,
            "fourier_scale_factor": total_fourier_scale,

            "external_innovations_used": innovations is not None,
            "additional_component_used": additional_component is not None
        }

        return df, info

    def generate_seasonal_unit_root_fourier(
        self,
        period=None,
        amplitude=None,
        noise_std=None,
        scale_factor=1.0,
        initial_std=0.2,
        num_harmonics=1,
        order_range=(0, 2),
        coef_range=(-0.4, 0.4),
        seasonal_strength_range=(0.8, 2.0),
        allowed_periods=None,
        min_cycles = 6, 
        max_attempts=1000
    ):
        """
        Special seasonal-unit-root + deterministic Fourier model.

        This function implements the advisor-requested special case:

            (1 - B^s) Y_t = F_t + u_t

        equivalently:

            Y_t - Y_{t-s} = F_t + u_t

        where
        -----
        F_t :
            deterministic Fourier seasonality with known period s

        u_t :
            stationary NON-SEASONAL ARMA process

        Model properties
        ----------------
        - seasonal unit root: D = 1
        - non-seasonal differencing: d = 0
        - seasonal AR order: P = 0
        - seasonal MA order: Q = 0

        Important
        ---------
        Seasonal differencing recovers:

            Delta_s Y_t = F_t + u_t

        Therefore the deterministic Fourier component is directly
        observable in the seasonally differenced representation.

        This is a project-specific special case and is kept separate
        from generate_deterministic_sarima() and generate_pure_sarima().
        """

        n = self.length
        t = np.arange(n)


        # =====================================================
        # SEASONAL PERIOD
        # =====================================================

        period, _ = self.choose_calendar_period(
            period=period,
            allowed_periods=allowed_periods,
            min_cycles=min_cycles
        )

        s = int(period)

        if n <= s:
            raise ValueError(
                "Series length must be greater than seasonal period."
            )

        # =====================================================
        # INNOVATION SCALE
        # =====================================================

        if noise_std is None:
            noise_std = np.random.uniform(
                0.1,
                0.20
            )

        # =====================================================
        # NON-SEASONAL STATIONARY ARMA ERROR
        #
        # phi(B) u_t = theta(B) epsilon_t
        # =====================================================

        arma_process = None

        for _ in range(max_attempts):

            ar_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            ma_order = np.random.randint(
                order_range[0],
                order_range[1] + 1
            )

            if ar_order > 0:

                ar_coefs = np.random.uniform(
                    coef_range[0],
                    coef_range[1],
                    ar_order
                )

            else:

                ar_coefs = np.array(
                    [],
                    dtype=float
                )

            if ma_order > 0:

                ma_coefs = np.random.uniform(
                    coef_range[0],
                    coef_range[1],
                    ma_order
                )

            else:

                ma_coefs = np.array(
                    [],
                    dtype=float
                )

            ar_poly = np.r_[
                1.0,
                -ar_coefs
            ]

            ma_poly = np.r_[
                1.0,
                ma_coefs
            ]

            candidate = ArmaProcess(
                ar_poly,
                ma_poly
            )

            if (
                candidate.isstationary
                and candidate.isinvertible
            ):
                arma_process = candidate
                break

        if arma_process is None:
            raise RuntimeError(
                "Could not generate valid stationary ARMA error process."
            )

        # =====================================================
        # GENERATE STOCHASTIC ERROR u_t
        # =====================================================

        burnin = max(
            200,
            8 * s
        )

        stochastic_error = (
            arma_process.generate_sample(
                nsample=n,
                burnin=burnin,
                scale=noise_std
            )
        )

        # =====================================================
        # INITIAL FOURIER
        # =====================================================

        if amplitude is None:
            amplitude = 1.0

        fourier = np.zeros(
            n
        )

        fourier_coefficients = []

        for k in range(
            1,
            num_harmonics + 1
        ):

            A_k = (
                amplitude
                * np.random.uniform(
                    0.5,
                    1.0
                )
                / k
            )

            B_k = (
                amplitude
                * np.random.uniform(
                    0.5,
                    1.0
                )
                / k
            )

            fourier += (
                A_k
                * np.sin(
                    2 * np.pi * k * t / s
                )
            )

            fourier += (
                B_k
                * np.cos(
                    2 * np.pi * k * t / s
                )
            )

            fourier_coefficients.append({
                "harmonic": k,
                "sin_coef": A_k,
                "cos_coef": B_k
            })

        # =====================================================
        # SCALE FACTOR
        # =====================================================

        fourier *= scale_factor

        for coef_info in fourier_coefficients:

            coef_info["sin_coef"] *= (
                scale_factor
            )

            coef_info["cos_coef"] *= (
                scale_factor
            )

        # =====================================================
        # FOURIER STRENGTH CALIBRATION
        #
        # Here u_t is stationary, so its standard deviation
        # is a sensible reference scale.
        # =====================================================

        stochastic_scale = np.std(
            stochastic_error
        )

        current_fourier_std = np.std(
            fourier
        )

        seasonal_strength = (
            np.random.uniform(
                seasonal_strength_range[0],
                seasonal_strength_range[1]
            )
        )

        target_fourier_std = (
            seasonal_strength
            * stochastic_scale
        )

        if (
            stochastic_scale > 1e-8
            and current_fourier_std > 1e-8
        ):

            fourier_rescale_factor = (
                target_fourier_std
                / current_fourier_std
            )

            fourier *= (
                fourier_rescale_factor
            )

            for coef_info in fourier_coefficients:

                coef_info["sin_coef"] *= (
                    fourier_rescale_factor
                )

                coef_info["cos_coef"] *= (
                    fourier_rescale_factor
                )

        else:

            fourier_rescale_factor = 1.0

        # =====================================================
        # SEASONALLY DIFFERENCED PROCESS
        #
        # W_t = F_t + u_t
        #
        # and:
        #
        # (1-B^s)Y_t = W_t
        # =====================================================

        seasonal_difference_source = (
            fourier
            + stochastic_error
        )

        # =====================================================
        # SEASONAL INTEGRATION
        #
        # Y_t = Y_{t-s} + W_t
        # =====================================================

        series = np.zeros(
            n
        )

        # Initial seasonal states
        series[:s] = np.random.normal(
            loc=0.0,
            scale=initial_std,
            size=s
        )

        for i in range(
            s,
            n
        ):

            series[i] = (
                series[i - s]
                + seasonal_difference_source[i]
            )

        # =====================================================
        # RECOVER SEASONAL DIFFERENCE
        # =====================================================

        seasonal_difference = np.full(
            n,
            np.nan
        )

        seasonal_difference[s:] = (
            series[s:]
            - series[:-s]
        )

        # =====================================================
        # DATAFRAME
        # =====================================================

        df = pd.DataFrame({

            "time":
                np.arange(n),

            "data":
                series,

            # Ground-truth deterministic component
            "fourier_component":
                fourier,

            # Ground-truth stochastic ARMA contribution
            "stochastic_error":
                stochastic_error,

            # Exact RHS used before seasonal integration
            "seasonal_difference_source":
                seasonal_difference_source,

            # Recovered from final Y_t
            "seasonal_difference":
                seasonal_difference,

            "stationary":
                np.zeros(n).astype(int),

            "seasonal":
                np.ones(n).astype(int),

            "seasonal_unit_root":
                np.ones(n).astype(int)
        })

        # =====================================================
        # METADATA
        # =====================================================

        info = {

            "type":
                "seasonal",

            "subtype":
                "SEASONAL_UNIT_ROOT_FOURIER",

            "periods":
                [s],

            "period_meanings": {
                s: self.get_period_meanings(s)
            },

            "seasonality_source":
                "fourier_in_seasonal_difference_equation",

            "stochastic_background":
                "ARMA",

            # ---------------------------------------------
            # DIFFERENCING
            # ---------------------------------------------

            "diff":
                0,

            "seasonal_diff":
                1,

            "unit_root":
                "none",

            "seasonal_unit_root":
                "seasonal_unit_root",

            # ---------------------------------------------
            # NON-SEASONAL ARMA
            # ---------------------------------------------

            "ar_order":
                ar_order,

            "ma_order":
                ma_order,

            "ar_coefs":
                ar_coefs,

            "ma_coefs":
                ma_coefs,

            # Deliberately no seasonal ARMA terms
            "seasonal_ar_order":
                0,

            "seasonal_ma_order":
                0,

            "seasonal_ar_coefs":
                np.array(
                    [],
                    dtype=float
                ),

            "seasonal_ma_coefs":
                np.array(
                    [],
                    dtype=float
                ),

            # ---------------------------------------------
            # FOURIER
            # ---------------------------------------------

            "num_harmonics":
                num_harmonics,

            "fourier_coefficients":
                fourier_coefficients,

            "seasonal_strength":
                seasonal_strength,

            "fourier_std":
                np.std(
                    fourier
                ),

            "stochastic_scale":
                stochastic_scale,

            "fourier_rescale_factor":
                fourier_rescale_factor,

            # ---------------------------------------------
            # OTHER
            # ---------------------------------------------

            "noise_std":
                noise_std,

            "initial_std":
                initial_std,

            "fourier_used":
                True
        }

        return df, info

    def generate_seasonality_from_base_series(
        self,
        kind=None,
        num_components=2,
        period=None
    ):
        """
        Generates seasonal base series from scratch.

        Main dataset seasonal types
        ---------------------------
        "single"
            -> deterministic single Fourier seasonality

        "multiple"
            -> deterministic multiple Fourier seasonality

        "sarma"
            -> deterministic Fourier seasonality
            + non-seasonal stationary ARMA background

            Y_t = F_t + U_t

            where:
                phi(B) U_t = theta(B) epsilon_t

        "sarima"
            -> deterministic Fourier seasonality
            + non-seasonal ARIMA background

            Y_t = F_t + Z_t

            where:
                phi(B)(1-B)^d Z_t
                    = theta(B) epsilon_t


        Additional experimental / reference types
        -----------------------------------------
        "pure_sarma"
            -> textbook stochastic SARMA

            phi(B) Phi(B^s) Y_t
                = theta(B) Theta(B^s) epsilon_t

        "pure_sarima"
            -> textbook stochastic SARIMA

            phi(B) Phi(B^s)
            (1-B)^d (1-B^s)^D Y_t
                =
            theta(B) Theta(B^s) epsilon_t

        "seasonal_unit_root"
            -> special deterministic Fourier
            seasonal-unit-root model

            (1-B^s)Y_t = F_t + u_t

            where u_t is a stationary non-seasonal ARMA process.


        Important
        ---------
        If kind is None, only the four MAIN dataset seasonal
        families are sampled:

            single
            multiple
            sarma
            sarima

        Pure stochastic and special seasonal-unit-root cases
        must be requested explicitly.
        """

        # =====================================================
        # DEFAULT DATASET SAMPLING
        # =====================================================

        if kind is None:
            kind = random.choice([
                "single",
                "multiple",
                "sarma",
                "sarima"
            ])

        # =====================================================
        # SINGLE FOURIER SEASONALITY
        # =====================================================

        if kind == "single":

            df, info = self.generate_single_seasonality(
                period=period,
                num_harmonics=1
            )

        # =====================================================
        # MULTIPLE FOURIER SEASONALITY
        # =====================================================

        elif kind == "multiple":

            periods = self.normalize_period_list(
                period
            )

            df, info = self.generate_multiple_seasonality(
                num_components=num_components,
                periods=periods,
                num_harmonics=1
            )

        # =====================================================
        # DETERMINISTIC SARMA
        #
        # Fourier + ARMA
        # =====================================================

        elif kind == "sarma":

            df, info = self.generate_deterministic_sarma(
                period=period,
                num_harmonics=1
            )

        # =====================================================
        # DETERMINISTIC SARIMA
        #
        # Fourier + ARIMA
        # =====================================================

        elif kind == "sarima":

            df, info = self.generate_deterministic_sarima(
                period=period,
                num_harmonics=1
            )

        # =====================================================
        # PURE STOCHASTIC SARMA
        # =====================================================

        elif kind == "pure_sarma":

            df, info = self.generate_pure_sarma(
                period=period
            )

        # =====================================================
        # PURE STOCHASTIC SARIMA
        # =====================================================

        elif kind == "pure_sarima":

            df, info = self.generate_pure_sarima(
                period=period
            )

        # =====================================================
        # SPECIAL SEASONAL-UNIT-ROOT + FOURIER CASE
        # =====================================================

        elif kind == "seasonal_unit_root":

            df, info = self.generate_seasonal_unit_root_fourier(
                period=period,
                num_harmonics=1
            )

        # =====================================================
        # INVALID KIND
        # =====================================================

        else:

            raise ValueError(
                "Invalid kind. Choose from: "
                "'single', 'multiple', 'sarma', 'sarima', "
                "'pure_sarma', 'pure_sarima', "
                "or 'seasonal_unit_root'."
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
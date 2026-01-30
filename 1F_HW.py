"""
One-Factor Hull-White Model Implementation

Implements analytical formulas from Vithanalage (2024):
- Zero-coupon bond pricing
- Caplet/Cap pricing
- Swaption pricing (Jamshidian decomposition)
- Model calibration to market volatilities

Model: dr(t) = [theta(t) - gamma * r(t)] dt + sigma * dW(t)

Reference: Vithanalage (2024), Chapter 3 & 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from typing import Optional, Tuple, Dict, List
import warnings


class HullWhite1F:
    """
    One-Factor Hull-White interest rate model
    
    Parameters:
        gamma: Mean reversion speed
        sigma: Volatility
        theta(t): Time-dependent drift (calibrated to initial curve)
    """
    
    def __init__(self, 
                 gamma: float = 0.1, 
                 sigma: float = 0.01,
                 yield_curve: 'EURYieldCurve' = None):
        """
        Initialize Hull-White 1F model
        
        Args:
            gamma: Mean reversion parameter
            sigma: Volatility parameter
            yield_curve: EURYieldCurve object for theta(t) calibration
        """
        self.gamma = gamma
        self.sigma = sigma
        self.yield_curve = yield_curve
        
        # Calibration results
        self.calibration_results = None
    
    def B(self, T: float) -> float:
        """
        Calculate B(0,T) function
        
        Formula (Equation 4.2 from paper):
        B(T) = (1 - exp(-gamma * T)) / gamma
        
        Args:
            T: Time to maturity
            
        Returns:
            B(0,T)
        """
        if abs(self.gamma) < 1e-8:
            return T
        return (1.0 - np.exp(-self.gamma * T)) / self.gamma
    
    def A(self, t: float, T: float, 
          Z_t: float, Z_T: float, f0t: float) -> float:
        """
        Calculate A(t,T) function
        
        Formula (from paper, line 84):
        A(t,T) = ln(Z(T)/Z(t)) + B(t,T)*f(0,t) - sigma^2/2 * B(t,T)^2 * B(2*gamma, t)
        
        Args:
            t: Current time
            T: Maturity time
            Z_t: Discount factor at t
            Z_T: Discount factor at T
            f0t: Instantaneous forward rate f(0,t)
            
        Returns:
            A(t,T)
        """
        B_tT = self.B(T - t)
        B_2gamma_t = self.B_vasicek(2 * self.gamma, t)
        
        A_tT = (np.log(Z_T / Z_t) + 
                B_tT * f0t - 
                (self.sigma**2 / 2) * B_tT**2 * B_2gamma_t)
        
        return A_tT
    
    @staticmethod
    def B_vasicek(gamma: float, T: float) -> float:
        """
        Vasicek B function (used in variance calculations)
        
        Formula (Equation 4.3 from paper):
        B(gamma, T) = (1 - exp(-gamma * T)) / gamma
        
        Args:
            gamma: Parameter
            T: Time
            
        Returns:
            B(gamma, T)
        """
        if abs(gamma) < 1e-8:
            return T
        return (1.0 - np.exp(-gamma * T)) / gamma
    
    def sigma_Z(self, T_O: float, T_B: float) -> float:
        """
        Calculate volatility of zero-coupon bond ratio
        
        Formula (Vasicek.SZ from paper):
        sigma_Z = sigma * B(T_B - T_O) * sqrt(B(2*gamma, T_O))
        
        Used in option pricing formulas
        
        Args:
            T_O: Option expiry
            T_B: Bond maturity
            
        Returns:
            Volatility sigma_Z
        """
        B_diff = self.B(T_B - T_O)
        B_2gamma = self.B_vasicek(2 * self.gamma, T_O)
        
        return self.sigma * B_diff * np.sqrt(B_2gamma)
    
    def price_zcb(self, t: float, T: float, r_t: float) -> float:
        """
        Price zero-coupon bond under Hull-White
        
        Formula (Equation 3.11 from paper):
        P(t,T) = exp(A(t,T) - B(t,T) * r(t))
        
        Args:
            t: Current time
            T: Maturity
            r_t: Short rate at time t
            
        Returns:
            Zero-coupon bond price
        """
        if self.yield_curve is None:
            raise ValueError("Yield curve required for ZCB pricing")
        
        Z_t = self.yield_curve.get_discount_factor(t)
        Z_T = self.yield_curve.get_discount_factor(T)
        f0t = self.yield_curve.get_forward_rate(t)
        
        A_tT = self.A(t, T, Z_t, Z_T, f0t)
        B_tT = self.B(T - t)
        
        return np.exp(A_tT - B_tT * r_t)
    
    def price_caplet(self, 
                     strike: float, 
                     T: float, 
                     delta: float,
                     Z_TO: float, 
                     Z_TB: float) -> float:
        """
        Price single caplet using Hull-White formula
        
        Formula (from paper, lines 92-93):
        Caplet = Z(T_O) * N(-d2) - Z(T_B)/K * N(-d1)
        
        where:
        K = 1 / (1 + strike * delta)
        d1 = [ln(Z(T_B)/(K*Z(T_O))) + sigma_Z^2/2] / sigma_Z
        d2 = d1 - sigma_Z
        
        Args:
            strike: Strike rate
            T: Caplet maturity
            delta: Accrual period (0.25 for quarterly)
            Z_TO: Discount factor to T-delta
            Z_TB: Discount factor to T
            
        Returns:
            Caplet price
        """
        T_O = T - delta
        T_B = T
        
        K = 1.0 / (1.0 + strike * delta)
        
        sigma_Z = self.sigma_Z(T_O, T_B)
        
        d1 = (np.log(Z_TB / (K * Z_TO)) + sigma_Z**2 / 2) / sigma_Z
        d2 = d1 - sigma_Z
        
        caplet_price = Z_TO * norm.cdf(-d2) - (Z_TB / K) * norm.cdf(-d1)
        
        return caplet_price
    
    def price_cap(self, 
                  strike: float, 
                  maturity: float, 
                  delta: float = 0.25) -> float:
        """
        Price cap as sum of caplets
        
        Formula (from paper, lines 93):
        Cap = sum of caplets from 2*delta to maturity
        
        Args:
            strike: Strike rate
            maturity: Cap maturity
            delta: Payment frequency
            
        Returns:
            Cap price
        """
        if self.yield_curve is None:
            raise ValueError("Yield curve required for cap pricing")
        
        # Caplet maturities (start from 2*delta)
        caplet_times = np.arange(2 * delta, maturity + delta, delta)
        
        cap_price = 0.0
        
        for T in caplet_times:
            T_O = T - delta
            T_B = T
            
            Z_TO = self.yield_curve.get_discount_factor(T_O)
            Z_TB = self.yield_curve.get_discount_factor(T_B)
            
            caplet_price = self.price_caplet(strike, T, delta, Z_TO, Z_TB)
            cap_price += caplet_price
        
        return cap_price
    
    def price_swaption_jamshidian(self,
                                   expiry: float,
                                   tenor: float,
                                   strike: float,
                                   delta: float = 0.25,
                                   option_type: str = 'payer') -> float:
        """
        Price swaption using Jamshidian decomposition
        
        Implements the method from paper (lines 84-85):
        1. Find critical rate r* such that swap value = strike
        2. Decompose into portfolio of bond options
        3. Sum option values
        
        Args:
            expiry: Swaption expiry (T_O)
            tenor: Underlying swap tenor
            strike: Strike rate
            delta: Payment frequency
            option_type: 'payer' or 'receiver'
            
        Returns:
            Swaption price
        """
        if self.yield_curve is None:
            raise ValueError("Yield curve required for swaption pricing")
        
        T_O = expiry
        T_B = expiry + tenor
        
        # Payment dates
        payment_dates = np.arange(T_O + delta, T_B + delta, delta)
        payment_dates = payment_dates[payment_dates > T_O]
        
        # Cash flows
        n_payments = len(payment_dates)
        cash_flows = np.full(n_payments, strike * delta)
        cash_flows[-1] += 1.0  # Principal at maturity
        
        # Get discount factors and forward rate at expiry
        Z_t = self.yield_curve.get_discount_factor(T_O)
        Z_T = np.array([self.yield_curve.get_discount_factor(t) for t in payment_dates])
        f0t = self.yield_curve.get_forward_rate(T_O)
        
        # Find critical rate r* (root finding)
        def swap_value(r_star):
            """Value of coupon bond at expiry"""
            A_tT = np.array([self.A(T_O, t, Z_t, Z_T[i], f0t) 
                            for i, t in enumerate(payment_dates)])
            B_tT = np.array([self.B(t - T_O) for t in payment_dates])
            
            bond_prices = np.exp(A_tT - B_tT * r_star)
            return np.sum(cash_flows * bond_prices) - 1.0
        
        try:
            r_star = brentq(swap_value, -1.0, 1.0, xtol=1e-15)
        except ValueError:
            warnings.warn("Could not find critical rate, using approximation")
            r_star = 0.0
        
        # Calculate A(t,T) and B(t,T) for each payment date
        A_tT = np.array([self.A(T_O, t, Z_t, Z_T[i], f0t) 
                        for i, t in enumerate(payment_dates)])
        B_tT = np.array([self.B(t - T_O) for t in payment_dates])
        
        # Strike prices for bond options
        K_i = np.exp(A_tT - B_tT * r_star)
        
        # Volatilities for each bond option
        sigma_Z_array = np.array([self.sigma_Z(T_O, t) for t in payment_dates])
        
        # Price each bond option
        d1 = (np.log(Z_T / (K_i * Z_t)) + sigma_Z_array**2 / 2) / sigma_Z_array
        d2 = d1 - sigma_Z_array
        
        if option_type == 'payer':
            # Call options on bonds
            option_values = Z_T * norm.cdf(d1) - K_i * Z_t * norm.cdf(d2)
        else:
            # Put options on bonds
            option_values = K_i * Z_t * norm.cdf(-d2) - Z_T * norm.cdf(-d1)
        
        # Swaption value
        swaption_price = np.sum(cash_flows * option_values)
        
        return swaption_price
    
    def calibrate_to_caps(self, 
                          cap_data: pd.DataFrame,
                          initial_guess: Tuple[float, float] = (0.1, 0.01),
                          bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.001, 2.0), (0.0001, 0.5))) -> Dict:
        """
        Calibrate gamma and sigma to market cap prices
        
        Minimizes sum of squared errors between model and market prices
        
        Args:
            cap_data: DataFrame with columns ['maturity', 'strike', 'market_price']
            initial_guess: (gamma, sigma) starting values
            bounds: Parameter bounds
            
        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("HULL-WHITE 1F CALIBRATION TO CAPS")
        print("="*60)
        
        def objective(params):
            """Objective function: sum of squared pricing errors"""
            gamma_trial, sigma_trial = params
            
            # Temporarily set parameters
            self.gamma = gamma_trial
            self.sigma = sigma_trial
            
            errors = []
            for _, row in cap_data.iterrows():
                maturity = row['maturity']
                strike = row['strike']
                market_price = row['market_price']
                
                try:
                    model_price = self.price_cap(strike, maturity)
                    error = (model_price - market_price)**2
                    errors.append(error)
                except:
                    errors.append(1e6)  # Penalty for failed pricing
            
            return np.sum(errors)
        
        # Optimization
        print("\nOptimizing parameters...")
        result = minimize(
            objective,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        # Set calibrated parameters
        self.gamma = result.x[0]
        self.sigma = result.x[1]
        
        # Calculate final errors
        model_prices = []
        market_prices = []
        errors = []
        
        for _, row in cap_data.iterrows():
            market_price = row['market_price']
            model_price = self.price_cap(row['strike'], row['maturity'])
            
            model_prices.append(model_price)
            market_prices.append(market_price)
            errors.append(model_price - market_price)
        
        # Store results
        self.calibration_results = {
            'gamma': self.gamma,
            'sigma': self.sigma,
            'rmse': np.sqrt(np.mean(np.array(errors)**2)),
            'mae': np.mean(np.abs(errors)),
            'success': result.success,
            'message': result.message,
            'model_prices': model_prices,
            'market_prices': market_prices,
            'errors': errors
        }
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"\nOptimized parameters:")
        print(f"  gamma = {self.gamma:.6f}")
        print(f"  sigma = {self.sigma:.6f}")
        print(f"\nFit quality:")
        print(f"  RMSE = {self.calibration_results['rmse']:.6f}")
        print(f"  MAE  = {self.calibration_results['mae']:.6f}")
        print(f"\nOptimization status: {result.message}")
        
        return self.calibration_results
    
    def calibrate_to_swaptions(self,
                                swaption_data: pd.DataFrame,
                                initial_guess: Tuple[float, float] = (0.1, 0.01),
                                bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.001, 2.0), (0.0001, 0.5))) -> Dict:
        """
        Calibrate gamma and sigma to market swaption prices
        
        Implements the method from paper (Section 4.2.4)
        
        Args:
            swaption_data: DataFrame with ['expiry', 'tenor', 'strike', 'market_price']
            initial_guess: (gamma, sigma) starting values
            bounds: Parameter bounds
            
        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("HULL-WHITE 1F CALIBRATION TO SWAPTIONS")
        print("="*60)
        
        def objective(params):
            """Objective function: sum of squared pricing errors"""
            gamma_trial, sigma_trial = params
            
            self.gamma = gamma_trial
            self.sigma = sigma_trial
            
            errors = []
            for _, row in swaption_data.iterrows():
                expiry = row['expiry']
                tenor = row['tenor']
                strike = row['strike']
                market_price = row['market_price']
                
                try:
                    model_price = self.price_swaption_jamshidian(
                        expiry, tenor, strike
                    )
                    error = (model_price - market_price)**2
                    errors.append(error)
                except:
                    errors.append(1e6)
            
            return np.sum(errors)
        
        # Optimization
        print("\nOptimizing parameters...")
        result = minimize(
            objective,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        # Set calibrated parameters
        self.gamma = result.x[0]
        self.sigma = result.x[1]
        
        # Calculate final errors
        model_prices = []
        market_prices = []
        errors = []
        
        for _, row in swaption_data.iterrows():
            market_price = row['market_price']
            model_price = self.price_swaption_jamshidian(
                row['expiry'], row['tenor'], row['strike']
            )
            
            model_prices.append(model_price)
            market_prices.append(market_price)
            errors.append(model_price - market_price)
        
        # Store results
        self.calibration_results = {
            'gamma': self.gamma,
            'sigma': self.sigma,
            'rmse': np.sqrt(np.mean(np.array(errors)**2)),
            'mae': np.mean(np.abs(errors)),
            'success': result.success,
            'message': result.message,
            'model_prices': model_prices,
            'market_prices': market_prices,
            'errors': errors
        }
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"\nOptimized parameters:")
        print(f"  gamma = {self.gamma:.6f}")
        print(f"  sigma = {self.sigma:.6f}")
        print(f"\nFit quality:")
        print(f"  RMSE = {self.calibration_results['rmse']:.6f}")
        print(f"  MAE  = {self.calibration_results['mae']:.6f}")
        print(f"\nOptimization status: {result.message}")
        
        return self.calibration_results
    
    def simulate_paths(self, 
                       T: float, 
                       n_steps: int, 
                       n_paths: int,
                       r0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate short rate paths using Euler discretization
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            r0: Initial rate (if None, use curve)
            
        Returns:
            (times, paths) where paths has shape (n_paths, n_steps+1)
        """
        if r0 is None:
            if self.yield_curve is None:
                r0 = 0.03
            else:
                r0 = self.yield_curve.get_forward_rate(0.01)
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0
        
        # Get theta(t) from yield curve
        if self.yield_curve is not None:
            theta_t = np.array([self.yield_curve.get_forward_rate(t) for t in times])
        else:
            theta_t = np.full(n_steps + 1, r0)
        
        # Simulate
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            drift = (theta_t[i] - self.gamma * paths[:, i]) * dt
            diffusion = self.sigma * dW
            
            paths[:, i + 1] = paths[:, i] + drift + diffusion
        
        return times, paths
    
    def plot_calibration_results(self, save_path: Optional[str] = None):
        """
        Plot calibration fit quality
        
        Args:
            save_path: Path to save figure
        """
        if self.calibration_results is None:
            print("No calibration results to plot")
            return
        
        model_prices = np.array(self.calibration_results['model_prices'])
        market_prices = np.array(self.calibration_results['market_prices'])
        errors = np.array(self.calibration_results['errors'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Model vs Market
        ax1 = axes[0]
        ax1.scatter(market_prices, model_prices, alpha=0.6, s=50)
        
        # 45-degree line
        min_price = min(market_prices.min(), model_prices.min())
        max_price = max(market_prices.max(), model_prices.max())
        ax1.plot([min_price, max_price], [min_price, max_price], 
                'r--', linewidth=2, label='Perfect fit')
        
        ax1.set_xlabel('Market Price', fontsize=12)
        ax1.set_ylabel('Model Price', fontsize=12)
        ax1.set_title('Model vs Market Prices', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pricing errors
        ax2 = axes[1]
        ax2.bar(range(len(errors)), errors, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Instrument Index', fontsize=12)
        ax2.set_ylabel('Pricing Error', fontsize=12)
        ax2.set_title('Calibration Errors', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def summary(self) -> str:
        """
        Return model summary
        
        Returns:
            Summary string
        """
        summary = f"""
Hull-White One-Factor Model
{'='*50}

Parameters:
  gamma (mean reversion) = {self.gamma:.6f}
  sigma (volatility)     = {self.sigma:.6f}

Yield Curve:
  Reference date = {self.yield_curve.reference_date if self.yield_curve else 'Not set'}
  
Calibration:
  Status = {'Calibrated' if self.calibration_results else 'Not calibrated'}
"""
        
        if self.calibration_results:
            summary += f"""
  RMSE = {self.calibration_results['rmse']:.6f}
  MAE  = {self.calibration_results['mae']:.6f}
"""
        
        return summary


# Usage example
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HULL-WHITE ONE-FACTOR MODEL")
    print("="*70 + "\n")
    
    # Create sample yield curve
    from yield_curve import create_sample_eur_curve
    
    print("[1] Creating sample EUR yield curve...")
    curve = create_sample_eur_curve()
    
    # Initialize Hull-White model
    print("\n[2] Initializing Hull-White 1F model...")
    hw_model = HullWhite1F(gamma=0.1, sigma=0.01, yield_curve=curve)
    
    print(hw_model.summary())
    
    # Test ZCB pricing
    print("\n[3] Testing zero-coupon bond pricing:")
    print("-" * 60)
    r_t = 0.03
    for T in [1, 5, 10]:
        zcb_price = hw_model.price_zcb(0, T, r_t)
        print(f"P(0,{T}Y) with r(0)={r_t:.2%}: {zcb_price:.6f}")
    
    # Test cap pricing
    print("\n[4] Testing cap pricing:")
    print("-" * 60)
    strike = 0.035
    for maturity in [2, 5, 10]:
        cap_price = hw_model.price_cap(strike, maturity)
        print(f"Cap {maturity}Y @ {strike:.2%}: {cap_price:.6f}")
    
    # Test swaption pricing
    print("\n[5] Testing swaption pricing:")
    print("-" * 60)
    expiry = 2
    tenor = 5
    strike = 0.035
    swaption_price = hw_model.price_swaption_jamshidian(expiry, tenor, strike)
    print(f"Swaption {expiry}Yx{tenor}Y @ {strike:.2%}: {swaption_price:.6f}")
    
    # Simulate paths
    print("\n[6] Simulating short rate paths...")
    print("-" * 60)
    times, paths = hw_model.simulate_paths(T=10, n_steps=100, n_paths=5)
    
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(times, paths[i, :] * 100, alpha=0.7, linewidth=1.5)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Short Rate (%)', fontsize=12)
    plt.title('Hull-White 1F Simulated Paths', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./hw1f_simulated_paths.png', dpi=300)
    plt.show()
    
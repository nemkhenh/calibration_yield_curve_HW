"""
Two-Factor Hull-White Model Implementation

Implements analytical formulas from Vithanalage (2024):
- Two-factor volatility structure
- Forward volatility calibration
- Jacobian-based optimization
- Correlation between factors

Model: 
  dr(t) = [theta(t) - gamma1*r1(t) - gamma2*r2(t)] dt + sigma1*dW1(t) + sigma2*dW2(t)
  where dW1(t) * dW2(t) = rho * dt

Reference: Vithanalage (2024), Chapter 4.2.3
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, least_squares
from typing import Optional, Tuple, Dict, List
import warnings


class HullWhite2F:
    """
    Two-Factor Hull-White interest rate model
    
    Parameters:
        gamma1, gamma2: Mean reversion speeds
        sigma1, sigma2: Volatilities
        rho: Correlation between factors
    """
    
    def __init__(self, 
                 gamma1: float = 0.1, 
                 gamma2: float = 0.5,
                 sigma1: float = 0.01,
                 sigma2: float = 0.015,
                 rho: float = 0.5,
                 yield_curve: 'EURYieldCurve' = None):
        """
        Initialize Hull-White 2F model
        
        Args:
            gamma1: Mean reversion for factor 1 (short-term)
            gamma2: Mean reversion for factor 2 (long-term)
            sigma1: Volatility for factor 1
            sigma2: Volatility for factor 2
            rho: Correlation between factors
            yield_curve: EURYieldCurve object
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.yield_curve = yield_curve
        
        # Calibration results
        self.calibration_results = None
    
    def B1(self, T: float) -> float:
        """
        Calculate B1(T) function for factor 1
        
        Formula: B1(T) = (1 - exp(-gamma1 * T)) / gamma1
        
        Args:
            T: Time to maturity
            
        Returns:
            B1(T)
        """
        if abs(self.gamma1) < 1e-8:
            return T
        return (1.0 - np.exp(-self.gamma1 * T)) / self.gamma1
    
    def B2(self, T: float) -> float:
        """
        Calculate B2(T) function for factor 2
        
        Formula: B2(T) = (1 - exp(-gamma2 * T)) / gamma2
        
        Args:
            T: Time to maturity
            
        Returns:
            B2(T)
        """
        if abs(self.gamma2) < 1e-8:
            return T
        return (1.0 - np.exp(-self.gamma2 * T)) / self.gamma2
    
    def B_vasicek(self, gamma: float, T: float) -> float:
        """
        Vasicek B function (used in variance calculations)
        
        Args:
            gamma: Parameter
            T: Time
            
        Returns:
            B(gamma, T)
        """
        if abs(gamma) < 1e-8:
            return T
        return (1.0 - np.exp(-gamma * T)) / gamma
    
    def DB_vasicek(self, gamma: float, T: float) -> float:
        """
        Derivative of Vasicek B function with respect to gamma
        
        Formula (from paper, lines 95-96):
        DB(gamma, T) = -(1-exp(-gamma*T))/gamma^2 + T*exp(-gamma*T)/gamma
        
        Args:
            gamma: Parameter
            T: Time
            
        Returns:
            dB/dgamma
        """
        if abs(gamma) < 1e-8:
            return -(T**2) / 2
        else:
            return (-(1 - np.exp(-gamma * T)) / gamma**2 + 
                    T * np.exp(-gamma * T) / gamma)
    
    def compute_forward_volatility_squared(self, 
                                           maturity: float, 
                                           delta: float = 0.25) -> float:
        """
        Calculate forward volatility squared for given maturity
        
        Formula (from paper, lines 95):
        sigma_f^2(T) = [(B1*sigma1)^2 * B(2*gamma1, T-delta) + 
                        (B2*sigma2)^2 * B(2*gamma2, T-delta) +
                        B1*B2*sigma1*sigma2*rho * B(gamma1+gamma2, T-delta)] / (T-delta)
        
        Args:
            maturity: Maturity in years
            delta: Payment frequency
            
        Returns:
            Forward volatility squared
        """
        T_O = maturity - delta
        
        if T_O <= 0:
            return 0.0
        
        B1_delta = self.B1(delta)
        B2_delta = self.B2(delta)
        
        B_2gamma1 = self.B_vasicek(2 * self.gamma1, T_O)
        B_2gamma2 = self.B_vasicek(2 * self.gamma2, T_O)
        B_gamma_sum = self.B_vasicek(self.gamma1 + self.gamma2, T_O)
        
        vol_sq = ((B1_delta * self.sigma1)**2 * B_2gamma1 +
                  (B2_delta * self.sigma2)**2 * B_2gamma2 +
                  B1_delta * B2_delta * self.sigma1 * self.sigma2 * self.rho * B_gamma_sum) / T_O
        
        return vol_sq
    
    def residual_function(self, 
                          params: np.ndarray, 
                          forward_vol_data: pd.DataFrame) -> np.ndarray:
        """
        Residual function for calibration
        
        Args:
            params: [gamma1, gamma2, sigma1, sigma2, rho]
            forward_vol_data: DataFrame with 'Maturities' and 'Forward Vol.sq'
            
        Returns:
            Array of residuals
        """
        self.gamma1, self.gamma2, self.sigma1, self.sigma2, self.rho = params
        
        maturities = forward_vol_data['Maturities'].values
        target_vol_sq = forward_vol_data['Forward Vol.sq'].values
        
        n = len(maturities)
        residuals = np.zeros(n)
        
        delta = 0.25
        
        for i in range(n):
            if i == 0:
                residuals[i] = 0.0
            else:
                model_vol_sq = self.compute_forward_volatility_squared(maturities[i], delta)
                residuals[i] = model_vol_sq - target_vol_sq[i]
        
        return residuals
    
    def jacobian_function(self, 
                          params: np.ndarray, 
                          forward_vol_data: pd.DataFrame) -> np.ndarray:
        """
        Jacobian matrix for calibration
        
        Args:
            params: [gamma1, gamma2, sigma1, sigma2, rho]
            forward_vol_data: DataFrame with forward volatility data
            
        Returns:
            Jacobian matrix (n x 5)
        """
        gamma1, gamma2, sigma1, sigma2, rho = params
        
        maturities = forward_vol_data['Maturities'].values
        n = len(maturities)
        delta = 0.25
        
        T_O = maturities - delta
        
        # Calculate B functions
        B1 = self.B_vasicek(gamma1, delta)
        B2 = self.B_vasicek(gamma2, delta)
        
        B1G1T = np.array([self.B_vasicek(2 * gamma1, t) for t in T_O])
        B2G2T = np.array([self.B_vasicek(2 * gamma2, t) for t in T_O])
        BG1G2T = np.array([self.B_vasicek(gamma1 + gamma2, t) for t in T_O])
        
        # Calculate derivatives
        B1D = self.DB_vasicek(gamma1, delta)
        B2D = self.DB_vasicek(gamma2, delta)
        
        B1G1TD = np.array([self.DB_vasicek(2 * gamma1, t) * 2 for t in T_O])
        B2G2TD = np.array([self.DB_vasicek(2 * gamma2, t) * 2 for t in T_O])
        BG1G2TD = np.array([self.DB_vasicek(gamma1 + gamma2, t) for t in T_O])
        
        # Initialize Jacobian
        Jacob = np.zeros((n, 5))
        
        # Partial derivatives (from paper lines 96-97)
        Jacob[:, 0] = ((2 * B1 * B1D * B1G1T + B1**2 * B1G1TD) * sigma1**2 +
                       (B1D * B2 * BG1G2T + B1 * B2 * BG1G2TD) * sigma1 * sigma2 * rho)
        
        Jacob[:, 1] = ((2 * B2 * B2D * B2G2T + B2**2 * B2G2TD) * sigma2**2 +
                       (B2D * B1 * BG1G2T + B1 * B2 * BG1G2TD) * sigma1 * sigma2 * rho)
        
        Jacob[:, 2] = B1**2 * B1G1T * 2 * sigma1 + B1 * B2 * BG1G2T * sigma2 * rho
        
        Jacob[:, 3] = B2**2 * B2G2T * 2 * sigma2 + B1 * B2 * BG1G2T * sigma1 * rho
        
        Jacob[:, 4] = B1 * B2 * BG1G2T * sigma1 * sigma2
        
        # Divide by T_O
        for i in range(n):
            if T_O[i] > 0:
                Jacob[i, :] = Jacob[i, :] / T_O[i]
            else:
                Jacob[i, :] = 0.0
        
        Jacob[0, :] = 0.0
        
        return Jacob
    
    def calibrate_to_forward_volatilities(self,
                                          forward_vol_data: pd.DataFrame,
                                          initial_guess: Tuple[float, float, float, float, float] = (0.05, 2.05, 0.4, 0.0004, 1.5),
                                          bounds: Tuple = ([-2, -2, 0.0001, 0.0001, -0.99], 
                                                          [2, 2, 1.0, 1.0, 0.99])) -> Dict:
        """
        Calibrate 2F Hull-White to forward volatility structure

        Uses nonlinear least squares with analytical Jacobian
        
        Args:
            forward_vol_data: DataFrame with 'Maturities' and 'Forward Vol.sq'
            initial_guess: (gamma1, gamma2, sigma1, sigma2, rho)
            bounds: Parameter bounds
            
        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("HULL-WHITE 2F CALIBRATION TO FORWARD VOLATILITIES")
        print("="*60)
        
        print(f"\nInitial parameters:")
        print(f"  gamma1 = {initial_guess[0]:.6f}")
        print(f"  gamma2 = {initial_guess[1]:.6f}")
        print(f"  sigma1 = {initial_guess[2]:.6f}")
        print(f"  sigma2 = {initial_guess[3]:.6f}")
        print(f"  rho    = {initial_guess[4]:.6f}")
        
        # Optimization with Jacobian
        print("\nOptimizing with analytical Jacobian...")
        
        result = least_squares(
            fun=lambda p: self.residual_function(p, forward_vol_data),
            x0=initial_guess,
            jac=lambda p: self.jacobian_function(p, forward_vol_data),
            bounds=bounds,
            method='trf',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=10000,
            verbose=0
        )
        
        # Set calibrated parameters
        self.gamma1 = result.x[0]
        self.gamma2 = result.x[1]
        self.sigma1 = result.x[2]
        self.sigma2 = result.x[3]
        self.rho = result.x[4]
        
        # Calculate final residuals
        residuals = self.residual_function(result.x, forward_vol_data)
        
        # Store results
        self.calibration_results = {
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'sigma1': self.sigma1,
            'sigma2': self.sigma2,
            'rho': self.rho,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals)),
            'success': result.success,
            'message': result.message,
            'residuals': residuals,
            'cost': result.cost
        }
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"\nOptimized parameters:")
        print(f"  gamma1 = {self.gamma1:.6f}")
        print(f"  gamma2 = {self.gamma2:.6f}")
        print(f"  sigma1 = {self.sigma1:.6f}")
        print(f"  sigma2 = {self.sigma2:.6f}")
        print(f"  rho    = {self.rho:.6f}")
        print(f"\nFit quality:")
        print(f"  RMSE      = {self.calibration_results['rmse']:.8f}")
        print(f"  MAE       = {self.calibration_results['mae']:.8f}")
        print(f"  Max Error = {self.calibration_results['max_error']:.8f}")
        print(f"  Cost      = {self.calibration_results['cost']:.8f}")
        print(f"\nOptimization status: {result.message}")
        
        return self.calibration_results
    
    def compute_volatility_term_structure(self, 
                                          maturities: np.ndarray,
                                          delta: float = 0.25) -> np.ndarray:
        """
        Calculate model-implied forward volatility term structure
        
        Args:
            maturities: Array of maturities
            delta: Payment frequency
            
        Returns:
            Array of forward volatilities
        """
        forward_vols = np.zeros(len(maturities))
        
        for i, mat in enumerate(maturities):
            vol_sq = self.compute_forward_volatility_squared(mat, delta)
            forward_vols[i] = np.sqrt(vol_sq) if vol_sq > 0 else 0.0
        
        return forward_vols
    
    def simulate_paths(self, 
                       T: float, 
                       n_steps: int, 
                       n_paths: int,
                       r1_0: Optional[float] = None,
                       r2_0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate two-factor short rate paths
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            r1_0: Initial value for factor 1
            r2_0: Initial value for factor 2
            
        Returns:
            (times, r1_paths, r2_paths)
        """
        if r1_0 is None:
            r1_0 = 0.02
        if r2_0 is None:
            r2_0 = 0.01
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        r1_paths = np.zeros((n_paths, n_steps + 1))
        r2_paths = np.zeros((n_paths, n_steps + 1))
        
        r1_paths[:, 0] = r1_0
        r2_paths[:, 0] = r2_0
        
        # Cholesky decomposition for correlation
        L = np.array([[1.0, 0.0],
                      [self.rho, np.sqrt(1 - self.rho**2)]])
        
        # Simulate
        for i in range(n_steps):
            # Independent normal draws
            Z = np.random.normal(0, 1, (n_paths, 2))
            
            # Correlated Brownian increments
            dW = np.sqrt(dt) * (Z @ L.T)
            
            # Factor 1
            drift1 = -self.gamma1 * r1_paths[:, i] * dt
            diffusion1 = self.sigma1 * dW[:, 0]
            r1_paths[:, i + 1] = r1_paths[:, i] + drift1 + diffusion1
            
            # Factor 2
            drift2 = -self.gamma2 * r2_paths[:, i] * dt
            diffusion2 = self.sigma2 * dW[:, 1]
            r2_paths[:, i + 1] = r2_paths[:, i] + drift2 + diffusion2
        
        return times, r1_paths, r2_paths
    
    def plot_calibration_results(self, 
                                 forward_vol_data: pd.DataFrame,
                                 save_path: Optional[str] = None):
        """
        Plot calibration fit quality
        
        Args:
            forward_vol_data: Original forward volatility data
            save_path: Path to save figure
        """
        if self.calibration_results is None:
            print("No calibration results to plot")
            return
        
        import matplotlib.pyplot as plt
        
        maturities = forward_vol_data['Maturities'].values
        market_vol_sq = forward_vol_data['Forward Vol.sq'].values
        
        # Calculate model volatilities
        model_vol_sq = np.array([
            self.compute_forward_volatility_squared(mat) 
            for mat in maturities
        ])
        
        residuals = self.calibration_results['residuals']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Model vs Market
        ax1 = axes[0]
        ax1.plot(maturities, np.sqrt(market_vol_sq) * 100, 
                'ro-', markersize=6, linewidth=2, label='Market')
        ax1.plot(maturities, np.sqrt(model_vol_sq) * 100, 
                'b^-', markersize=6, linewidth=2, label='Model (2F HW)')
        ax1.set_xlabel('Maturity (years)', fontsize=12)
        ax1.set_ylabel('Forward Volatility (%)', fontsize=12)
        ax1.set_title('Forward Volatility Term Structure', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2 = axes[1]
        ax2.bar(maturities, residuals * 100, alpha=0.6, width=0.2)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Maturity (years)', fontsize=12)
        ax2.set_ylabel('Residual (%)', fontsize=12)
        ax2.set_title('Calibration Residuals', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_simulated_paths(self, 
                            T: float = 10, 
                            n_steps: int = 100, 
                            n_paths: int = 5,
                            save_path: Optional[str] = None):
        """
        Plot simulated paths for both factors
        
        Args:
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        times, r1_paths, r2_paths = self.simulate_paths(T, n_steps, n_paths)
        
        # Total rate
        r_total = r1_paths + r2_paths
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Factor 1
        ax1 = axes[0, 0]
        for i in range(n_paths):
            ax1.plot(times, r1_paths[i, :] * 100, alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Time (years)', fontsize=11)
        ax1.set_ylabel('Rate (%)', fontsize=11)
        ax1.set_title('Factor 1 (Short-term)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Factor 2
        ax2 = axes[0, 1]
        for i in range(n_paths):
            ax2.plot(times, r2_paths[i, :] * 100, alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('Time (years)', fontsize=11)
        ax2.set_ylabel('Rate (%)', fontsize=11)
        ax2.set_title('Factor 2 (Long-term)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Total rate
        ax3 = axes[1, 0]
        for i in range(n_paths):
            ax3.plot(times, r_total[i, :] * 100, alpha=0.7, linewidth=1.5)
        ax3.set_xlabel('Time (years)', fontsize=11)
        ax3.set_ylabel('Rate (%)', fontsize=11)
        ax3.set_title('Total Short Rate (r1 + r2)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Distribution at final time
        ax4 = axes[1, 1]
        ax4.hist(r_total[:, -1] * 100, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Rate (%)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title(f'Distribution at T={T}Y', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
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
Hull-White Two-Factor Model
{'='*50}

Parameters:
  gamma1 (mean reversion 1) = {self.gamma1:.6f}
  gamma2 (mean reversion 2) = {self.gamma2:.6f}
  sigma1 (volatility 1)     = {self.sigma1:.6f}
  sigma2 (volatility 2)     = {self.sigma2:.6f}
  rho (correlation)         = {self.rho:.6f}

Yield Curve:
  Reference date = {self.yield_curve.reference_date if self.yield_curve else 'Not set'}
  
Calibration:
  Status = {'Calibrated' if self.calibration_results else 'Not calibrated'}
"""
        
        if self.calibration_results:
            summary += f"""
  RMSE      = {self.calibration_results['rmse']:.8f}
  MAE       = {self.calibration_results['mae']:.8f}
  Max Error = {self.calibration_results['max_error']:.8f}
"""
        
        return summary


# Usage example
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HULL-WHITE TWO-FACTOR MODEL")
    print("="*70 + "\n")
    
    # Create sample yield curve
    from yield_curve import create_sample_eur_curve
    
    print("[1] Creating sample EUR yield curve...")
    curve = create_sample_eur_curve()
    
    # Initialize Hull-White 2F model
    print("\n[2] Initializing Hull-White 2F model...")
    hw2f_model = HullWhite2F(
        gamma1=0.05, 
        gamma2=2.05,
        sigma1=0.4,
        sigma2=0.0004,
        rho=0.5,
        yield_curve=curve
    )
    
    print(hw2f_model.summary())
    
    # Create synthetic forward volatility data for testing
    print("\n[3] Creating synthetic forward volatility data...")
    maturities = np.arange(0.25, 30.25, 0.25)
    
    # Synthetic forward vol squared (decreasing with maturity)
    forward_vol_sq = 0.01 * np.exp(-0.1 * maturities) + 0.001
    
    forward_vol_data = pd.DataFrame({
        'Maturities': maturities,
        'Forward Vol.sq': forward_vol_sq
    })
    
    print(f"  {len(maturities)} maturity points")
    
    # Calibrate
    print("\n[4] Calibrating to forward volatilities...")
    results = hw2f_model.calibrate_to_forward_volatilities(forward_vol_data)
    
    # Plot results
    print("\n[5] Plotting calibration results...")
    hw2f_model.plot_calibration_results(forward_vol_data, 
                                        save_path='./hw2f_calibration.png')
    
    # Simulate paths
    print("\n[6] Simulating two-factor paths...")
    hw2f_model.plot_simulated_paths(T=10, n_steps=100, n_paths=5,
                                    save_path='./hw2f_simulated_paths.png')

    

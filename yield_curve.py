"""
EUR Yield Curve Construction and Manipulation Module

Implements:
- Cubic spline interpolation (as in the paper)
- Bootstrap of discount factors from swap rates
- Calculation of instantaneous forward rates f(0,t)
- EUR market conventions (ACT/360, annual payments)

Reference: Vithanalage (2024) - Calibration of Interest Rate Models
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import warnings


class EURYieldCurve:
    """
    Class for building and manipulating EUR yield curves
    
    Follows the research paper methodology with EUR adaptations
    """
    
    # EUR market conventions
    DAY_COUNT = 'ACT/360'
    PAYMENT_FREQUENCY = 1  # Annual for EUR swaps (vs semi-annual USD)
    DAYS_PER_YEAR = 360
    
    def __init__(self, 
                 reference_date: str,
                 swap_rates: pd.DataFrame = None,
                 interpolation_method: str = 'cubic'):
        """
        Initialize EUR yield curve
        
        Args:
            reference_date: Reference date (format 'YYYY-MM-DD')
            swap_rates: DataFrame with columns 'maturity_years' and 'swap_rate'
            interpolation_method: 'cubic' (default) or 'linear'
        """
        self.reference_date = reference_date
        self.interpolation_method = interpolation_method
        
        # Raw data
        self.market_maturities = None
        self.market_swap_rates = None
        
        # Interpolated curves
        self.maturities = None
        self.swap_rates = None
        self.discount_factors = None
        self.zero_rates = None
        self.forward_rates = None
        
        # Interpolators
        self._swap_interpolator = None
        self._df_interpolator = None
        self._forward_interpolator = None
        
        if swap_rates is not None:
            self.build_curve(swap_rates)
    
    def build_curve(self, swap_rates_df: pd.DataFrame):
        """
        Build complete curve from swap rates
        
        Pipeline:
        1. Interpolate swap rates (cubic spline)
        2. Bootstrap discount factors
        3. Calculate zero rates
        4. Calculate instantaneous forward rates f(0,t)
        
        Args:
            swap_rates_df: DataFrame with 'maturity_years' and 'swap_rate'
        """
        print("\n" + "="*60)
        print("EUR YIELD CURVE CONSTRUCTION")
        print("="*60)
        
        # Step 1: Load market data
        self.market_maturities = swap_rates_df['maturity_years'].values
        self.market_swap_rates = swap_rates_df['swap_rate'].values / 100  # To decimal
        
        print(f"\nMarket data loaded:")
        print(f"  - {len(self.market_maturities)} market points")
        print(f"  - Maturities: {self.market_maturities.min():.2f}Y to {self.market_maturities.max():.0f}Y")
        
        # Step 2: Interpolate swap rates
        print(f"\nInterpolating swap rates (method: {self.interpolation_method})...")
        self._interpolate_swap_rates()
        
        # Step 3: Bootstrap discount factors
        print(f"Bootstrapping discount factors...")
        self._bootstrap_discount_factors()
        
        # Step 4: Calculate zero rates
        print(f"Calculating zero rates...")
        self._compute_zero_rates()
        
        # Step 5: Calculate forward rates
        print(f"Calculating instantaneous forward rates f(0,t)...")
        self._compute_forward_rates()
        
        print("\n" + "="*60)
        print("EUR CURVE CONSTRUCTION COMPLETE")
        print("="*60)
        
        self._print_curve_summary()
    
    def _interpolate_swap_rates(self):
        """
        Interpolate swap rates with cubic spline
        
        Replicates R code from paper:
        model = splinefun(swap_curve$Maturity, swap_curve$Yield, method="natural")
        """
        # Fine grid for interpolation (quarterly = 0.25 years)
        delta = 0.25
        max_maturity = min(50, self.market_maturities.max())
        self.maturities = np.arange(delta, max_maturity + delta, delta)
        
        if self.interpolation_method == 'cubic':
            # Natural cubic spline (as in R with method="natural")
            self._swap_interpolator = CubicSpline(
                self.market_maturities,
                self.market_swap_rates,
                bc_type='natural'  # Natural boundary conditions
            )
        else:
            # Linear interpolation
            self._swap_interpolator = interp1d(
                self.market_maturities,
                self.market_swap_rates,
                kind='linear',
                fill_value='extrapolate'
            )
        
        # Interpolated swap rates
        self.swap_rates = self._swap_interpolator(self.maturities)
        
        print(f"  {len(self.maturities)} points interpolated")
    
    def _bootstrap_discount_factors(self):
        """
        Bootstrap discount factors from swap rates
        
        Replicates R code from paper (lines 82-87):
        
        For EUR swap with annual payments:
        SwapRate = (1 - DF(T)) / sum(DF(Ti) * Delta)
        
        Therefore: DF(T) = (1 - SwapRate * Delta * sum(DF(Ti))) / (1 + SwapRate * Delta)
        """
        n = len(self.maturities)
        self.discount_factors = np.zeros(n)
        
        delta = self.maturities[1] - self.maturities[0]  # 0.25 for quarterly
        
        # First discount factor (short term)
        # DF(T1) = 1 / (1 + rate * T1)
        self.discount_factors[0] = 1.0 / (1.0 + self.swap_rates[0] * delta)
        
        # Iterative bootstrap for other maturities
        for i in range(1, n):
            swap_rate = self.swap_rates[i]
            
            # Sum of previous discount factors
            sum_df = np.sum(self.discount_factors[:i])
            
            # Bootstrap formula (EUR annual convention)
            numerator = 1.0 - swap_rate * delta * sum_df
            denominator = 1.0 + swap_rate * delta
            
            self.discount_factors[i] = numerator / denominator
        
        print(f"  Discount factors bootstrapped")
        print(f"    - DF(1Y) = {self.get_discount_factor(1.0):.6f}")
        print(f"    - DF(5Y) = {self.get_discount_factor(5.0):.6f}")
        print(f"    - DF(10Y) = {self.get_discount_factor(10.0):.6f}")
    
    def _compute_zero_rates(self):
        """
        Calculate zero rates from discount factors
        
        Zero rate: r(T) = -ln(DF(T)) / T
        """
        # Avoid division by zero
        self.zero_rates = -np.log(self.discount_factors) / self.maturities
        
        print(f"  Zero rates calculated")
    
    def _compute_forward_rates(self):
        """
        Calculate instantaneous forward rates f(0,t)
        
        Replicates R code from paper (lines 88-90):
        rates = -log(discount_factors) / Maturities
        model1 = splinefun(Maturities, rates, method="natural")
        ratesdash = model1(Maturities, 1)  # First derivative
        f0t = rates + Maturities * ratesdash
        
        Formula: f(0,t) = r(t) + t * dr(t)/dt
        where r(t) is the zero rate
        """
        # Interpolator for zero rates
        zero_interpolator = CubicSpline(
            self.maturities,
            self.zero_rates,
            bc_type='natural'
        )
        
        # Derivative of zero rates
        zero_rates_derivative = zero_interpolator(self.maturities, 1)
        
        # Instantaneous forward rate
        self.forward_rates = self.zero_rates + self.maturities * zero_rates_derivative
        
        # Create interpolator for forward rates
        self._forward_interpolator = CubicSpline(
            self.maturities,
            self.forward_rates,
            bc_type='natural'
        )
        
        print(f"  Instantaneous forward rates f(0,t) calculated")
        print(f"    - f(0,1Y) = {self.get_forward_rate(1.0)*100:.3f}%")
        print(f"    - f(0,5Y) = {self.get_forward_rate(5.0)*100:.3f}%")
        print(f"    - f(0,10Y) = {self.get_forward_rate(10.0)*100:.3f}%")
    
    def get_discount_factor(self, maturity: float) -> float:
        """
        Return discount factor for given maturity
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Discount factor DF(0,T)
        """
        if self._df_interpolator is None:
            self._df_interpolator = CubicSpline(
                self.maturities,
                self.discount_factors,
                bc_type='natural'
            )
        
        return float(self._df_interpolator(maturity))
    
    def get_forward_rate(self, maturity: float) -> float:
        """
        Return instantaneous forward rate f(0,t)
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Forward rate f(0,t)
        """
        if self._forward_interpolator is None:
            self._forward_interpolator = CubicSpline(
                self.maturities,
                self.forward_rates,
                bc_type='natural'
            )
        
        return float(self._forward_interpolator(maturity))
    
    def get_zero_rate(self, maturity: float) -> float:
        """
        Return zero rate for given maturity
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Zero rate r(0,T)
        """
        df = self.get_discount_factor(maturity)
        return -np.log(df) / maturity
    
    def get_swap_rate(self, maturity: float) -> float:
        """
        Return interpolated swap rate
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Swap rate
        """
        return float(self._swap_interpolator(maturity))
    
    def compute_forward_swap_rate(self, 
                                   start: float, 
                                   end: float, 
                                   delta: float = 0.25) -> float:
        """
        Calculate forward swap rate between two dates
        
        Used for swaption pricing
        
        Args:
            start: Start date (years)
            end: End date (years)
            delta: Payment frequency (0.25 = quarterly)
            
        Returns:
            Forward swap rate
        """
        # Payment dates
        payment_dates = np.arange(start + delta, end + delta, delta)
        
        # Discount factors
        df_start = self.get_discount_factor(start)
        df_payments = np.array([self.get_discount_factor(t) for t in payment_dates])
        
        # Annuity
        annuity = np.sum(df_payments * delta)
        
        # Forward swap rate
        forward_swap = (df_start - df_payments[-1]) / annuity
        
        return forward_swap
    
    def compute_annuity(self, 
                        start: float, 
                        end: float, 
                        delta: float = 0.25) -> float:
        """
        Calculate swap annuity
        
        Annuity = sum(DF(Ti) * delta)
        
        Args:
            start: Start date
            end: End date
            delta: Payment frequency
            
        Returns:
            Annuity
        """
        payment_dates = np.arange(start + delta, end + delta, delta)
        df_payments = np.array([self.get_discount_factor(t) for t in payment_dates])
        
        return np.sum(df_payments * delta)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export complete curve to DataFrame
        
        Returns:
            DataFrame with all curves
        """
        df = pd.DataFrame({
            'maturity': self.maturities,
            'swap_rate': self.swap_rates * 100,
            'discount_factor': self.discount_factors,
            'zero_rate': self.zero_rates * 100,
            'forward_rate': self.forward_rates * 100,
            'reference_date': self.reference_date
        })
        
        return df
    
    def plot_curves(self, save_path: Optional[str] = None):
        """
        Visualize all curves (like Figures 5.11 and 5.13 from paper)
        
        Args:
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'EUR Yield Curves - {self.reference_date}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Swap rates (market vs interpolated)
        ax1 = axes[0, 0]
        ax1.plot(self.market_maturities, self.market_swap_rates * 100, 
                'ro', markersize=8, label='Market')
        ax1.plot(self.maturities, self.swap_rates * 100, 
                'b-', linewidth=2, label='Interpolated (Cubic Spline)')
        ax1.set_xlabel('Maturity (years)', fontsize=11)
        ax1.set_ylabel('Swap Rate (%)', fontsize=11)
        ax1.set_title('EUR Swap Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Discount factors
        ax2 = axes[0, 1]
        ax2.plot(self.maturities, self.discount_factors, 
                'g-', linewidth=2)
        ax2.set_xlabel('Maturity (years)', fontsize=11)
        ax2.set_ylabel('Discount Factor', fontsize=11)
        ax2.set_title('Discount Factors', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Zero rates
        ax3 = axes[1, 0]
        ax3.plot(self.maturities, self.zero_rates * 100, 
                'purple', linewidth=2)
        ax3.set_xlabel('Maturity (years)', fontsize=11)
        ax3.set_ylabel('Zero Rate (%)', fontsize=11)
        ax3.set_title('Zero-Coupon Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Instantaneous forward rates
        ax4 = axes[1, 1]
        ax4.plot(self.maturities, self.forward_rates * 100, 
                'orange', linewidth=2, label='f(0,t)')
        ax4.plot(self.maturities, self.zero_rates * 100, 
                'purple', linewidth=1, linestyle='--', alpha=0.5, label='r(0,t)')
        ax4.set_xlabel('Maturity (years)', fontsize=11)
        ax4.set_ylabel('Rate (%)', fontsize=11)
        ax4.set_title('Instantaneous Forward Rates', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved: {save_path}")
        
        plt.show()
    
    def plot_interpolation_comparison(self, save_path: Optional[str] = None):
        """
        Visualize interpolation quality (like Figure 5.13 from paper)
        
        Args:
            save_path: Path to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Market points
        ax.plot(self.market_maturities, self.market_swap_rates * 100,
               'ro', markersize=10, label='Market data', zorder=3)
        
        # Interpolated curve
        ax.plot(self.maturities, self.swap_rates * 100,
               'b-', linewidth=2, label='Cubic Spline Interpolation', zorder=2)
        
        # Interpolated points
        ax.plot(self.maturities, self.swap_rates * 100,
               'g.', markersize=4, alpha=0.5, label='Interpolated points', zorder=1)
        
        ax.set_xlabel('Maturity (years)', fontsize=12)
        ax.set_ylabel('Swap Rate (%)', fontsize=12)
        ax.set_title('Cubic Spline Interpolation - EUR Swap Rates', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _print_curve_summary(self):
        """Display curve summary"""
        print("\nCURVE SUMMARY:")
        print("-" * 60)
        
        # Key points
        key_maturities = [0.25, 1, 2, 5, 10, 20, 30]
        
        print(f"\n{'Maturity':<12} {'Swap':<10} {'Zero':<10} {'Forward':<10} {'DF':<10}")
        print("-" * 60)
        
        for mat in key_maturities:
            if mat <= self.maturities.max():
                swap = self.get_swap_rate(mat) * 100
                zero = self.get_zero_rate(mat) * 100
                forward = self.get_forward_rate(mat) * 100
                df = self.get_discount_factor(mat)
                
                print(f"{mat:>6.2f}Y     {swap:>6.3f}%    {zero:>6.3f}%    "
                      f"{forward:>6.3f}%    {df:>6.5f}")
    
    def get_curve_at_maturities(self, target_maturities: np.ndarray) -> pd.DataFrame:
        """
        Return curve values at specific maturities
        
        Useful for model calibration
        
        Args:
            target_maturities: Array of maturities in years
            
        Returns:
            DataFrame with all curves at requested maturities
        """
        data = {
            'maturity': target_maturities,
            'swap_rate': [self.get_swap_rate(t) for t in target_maturities],
            'discount_factor': [self.get_discount_factor(t) for t in target_maturities],
            'zero_rate': [self.get_zero_rate(t) for t in target_maturities],
            'forward_rate': [self.get_forward_rate(t) for t in target_maturities]
        }
        
        return pd.DataFrame(data)
    
    def compute_par_rate(self, maturity: float, delta: float = 0.25) -> float:
        """
        Calculate par rate (coupon rate that makes price = 100)
        
        Args:
            maturity: Maturity in years
            delta: Payment frequency
            
        Returns:
            Par rate
        """
        payment_dates = np.arange(delta, maturity + delta, delta)
        df_payments = np.array([self.get_discount_factor(t) for t in payment_dates])
        
        # Par rate = (1 - DF(T)) / sum(DF(Ti) * delta)
        par_rate = (1.0 - df_payments[-1]) / np.sum(df_payments * delta)
        
        return par_rate
    
    def save_curve(self, filepath: str):
        """
        Save curve to CSV
        
        Args:
            filepath: Output file path
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Curve saved: {filepath}")
    
    @classmethod
    def load_curve(cls, filepath: str) -> 'EURYieldCurve':
        """
        Load curve from CSV
        
        Args:
            filepath: File path
            
        Returns:
            EURYieldCurve instance
        """
        df = pd.read_csv(filepath)
        
        # Create instance
        curve = cls(reference_date=df['reference_date'].iloc[0])
        
        # Load data
        curve.maturities = df['maturity'].values
        curve.swap_rates = df['swap_rate'].values / 100
        curve.discount_factors = df['discount_factor'].values
        curve.zero_rates = df['zero_rate'].values / 100
        curve.forward_rates = df['forward_rate'].values / 100
        
        # Recreate interpolators
        curve._swap_interpolator = CubicSpline(
            curve.maturities, curve.swap_rates, bc_type='natural'
        )
        curve._df_interpolator = CubicSpline(
            curve.maturities, curve.discount_factors, bc_type='natural'
        )
        curve._forward_interpolator = CubicSpline(
            curve.maturities, curve.forward_rates, bc_type='natural'
        )
        
        print(f"Curve loaded: {filepath}")
        
        return curve


# Utility function for quick testing
def create_sample_eur_curve():
    """
    Create sample EUR curve for testing
    
    Returns:
        EURYieldCurve instance
    """
    # Synthetic data
    swap_data = pd.DataFrame({
        'maturity_years': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        'swap_rate': [3.85, 3.80, 3.70, 3.50, 3.40, 3.30, 3.35, 3.40, 3.45, 3.48, 3.50]
    })
    
    # Build curve
    curve = EURYieldCurve(
        reference_date='2024-06-30',
        swap_rates=swap_data,
        interpolation_method='cubic'
    )
    
    return curve


# Usage example
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" EUR YIELD CURVE MODULE - INTERPOLATION & BOOTSTRAP")
    print("="*70 + "\n")
    
    # Create sample curve
    print("[1] Creating sample EUR curve...\n")
    curve = create_sample_eur_curve()
    
    # Test data access
    print("\n[2] Testing data access:")
    print("-" * 60)
    
    test_maturities = [1, 5, 10]
    for mat in test_maturities:
        print(f"\nMaturity {mat}Y:")
        print(f"  - Swap rate:       {curve.get_swap_rate(mat)*100:.4f}%")
        print(f"  - Zero rate:       {curve.get_zero_rate(mat)*100:.4f}%")
        print(f"  - Forward rate:    {curve.get_forward_rate(mat)*100:.4f}%")
        print(f"  - Discount factor: {curve.get_discount_factor(mat):.6f}")
    
    # Test forward swap rate
    print("\n[3] Testing forward swap rate:")
    print("-" * 60)
    fwd_swap = curve.compute_forward_swap_rate(start=2, end=7)
    print(f"Forward swap 2Yx5Y: {fwd_swap*100:.4f}%")
    
    # Export DataFrame
    print("\n[4] Export to DataFrame:")
    print("-" * 60)
    df = curve.to_dataframe()
    print(df.head(10))
    
    # Visualization
    print("\n[5] Generating plots...")
    print("-" * 60)
    curve.plot_curves(save_path='./eur_yield_curves.png')
    curve.plot_interpolation_comparison(save_path='./eur_interpolation.png')
    
    # Save
    print("\n[6] Saving curve...")
    print("-" * 60)
    curve.save_curve('./data/eur_yield_curve.csv')

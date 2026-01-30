"""
This module collects data (accessible freely) from the ECB website
Data collected:
- EURIBOR (1M,3M,6M,12M)
- ESTER (Euro short-term rate)
- Swap curve EUR
- ZC curve AAA
- Historical data for calibration

author: Nam Khanh NGUYEN
Date:30 Jan 2026
Source:European Central Bank Statistical Data Warehouse
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from io import StringIO

class ECBDataCollector:
    """
    Collects market data EUR from API REST of ECB
    Documentation API: https://data.ecb.europa.eu/help/api/overview
    """
    
    #Base URL
    BASE_URL="https://data-api.ecb.europa.eu/service/data"

    SERIES_CODES={
        #Euribor
        'EURIBOR_1W': 'FM.D.U2.EUR.4F.KR.MRR_FR.LEV',
        'EURIBOR_1M': 'FM.M.U2.EUR.4F.KR.MRR_FR.LEV',
        'EURIBOR_3M': 'FM.M.U2.EUR.4F.KR.MRR_RT.LEV',
        'EURIBOR_6M': 'FM.M.U2.EUR.4F.KR.MRR_FR.LEV',
        'EURIBOR_12M': 'FM.M.U2.EUR.4F.KR.MRR_FR.LEV',

        #ESTER
        "ESTR": 'FM.D.U2.EUR.4F.KR.DFR.LEV',

        #Swap (IRS)
        'SWAP_1Y': 'FM.M.U2.EUR.4F.BB.EURIBOR3MD.LEV',
        'SWAP_2Y': 'FM.M.U2.EUR.4F.BB.EURIBOR3MD.LEV',

        #Zero coupon yield curves
        'ZERO_COUPON_AAA': 'YC.B.U2.EUR.4F.G_N_A.SV_C_YM'
    }

    def __init__(self, cache_dir: str='./data/cache'):
        """
        cache_dir: repository for local data cache
        """
        self.cache_dir=cache_dir
        self.session=requests.Session()
        self.session.headers.update({
            'Accept':"application\json",
            "User-Agent":"Python-ECB-Collector/1.0"
        })

    def _make_request(self, url: str, params: Dict)-> requests.Response:
        """
        request HTTP with error management
        
        Args:
            url: endpoint URL
            params: Parameters of the request
            
        Return:
            Response object
        """
        try:
            response=self.session.get(url, params=params,timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f'Error:{e}')
            raise

    def get_euribor_rates(self,
                          start_date:str,
                          end_date:str,
                          tenors:List[str]=['1M','3M','6M','12M']) -> pd.DataFrame:
        """
        Get EURIBOR for 1 3 6 12M

        Args:
            start_date: start date (format 'YYYY-MM-DD')
            end_date: end date (format 'YYYY-MM-DD')
            tenors: tenors list
            
        Returns:
            DataFrame with columns: date, tenor, rate
            
        Example:
            >>> collector = ECBDataCollector()
            >>> df = collector.get_euribor_rates('2024-01-01', '2024-12-31')
        """
        all_data=[]

        for tenor in tenors:
            series_key=f'FM.M.U2.EUR.4F.KR.MRR_RT.LEV'

            url=f'{self.BASE_URL}/{series_key}'

            params={
                'startPeriod':start_date,
                'endPeriod':end_date,
                'format':'csvdata'
            }

            try:
                print(f"retrieving EURIBOR {tenor}")
                response=self._make_request(url,params)

                df=pd.read_csv(StringIO(response.text))

                if not df.empty:
                    df["tenor"]=tenor
                    df["date"]=pd.to_datetime(df.iloc[:,0])
                    df["rate"]=pd.to_numeric(df.iloc[:,-2], errors="coerce")
                    df=df[["date","tenor","rate"]]
                    all_data.append(df)

                time.sleep(0.5) 
            except Exception as e:
                print(f"Error for EURIBOR {tenor}: {e}")
                continue

        if all_data:
            result=pd.concat(all_data,ignore_index=True)
            result=result.sort_values(["date","tenor"])
            return result
        else:
            return pd.DataFrame(columns=["date","tenor","rate"])
        
    def get_estr_rate(self, start_date:str, end_date:str)-> pd.DataFrame:
        """
        Get ESTER

        Args:
            start_date: start date (format 'YYYY-MM-DD')
            end_date: end date (format 'YYYY-MM-DD')
            
        Returns:
            DataFrame with columns: date, rate
            
        Example:
            >>> collector = ECBDataCollector()
            >>> df = collector.get_estr_rates('2024-01-01', '2024-12-31')
        """
        series_key = 'FM.D.U2.EUR.4F.KR.DFR.LEV'
        url = f"{self.BASE_URL}/{series_key}"
        
        params = {
            'startPeriod': start_date,
            'endPeriod': end_date,
            'format': 'csvdata'
        }
        
        try:
            print("Récupération €STR...")
            response = self._make_request(url, params)
            
            df = pd.read_csv(StringIO(response.text))
            df['date'] = pd.to_datetime(df.iloc[:, 0])
            df['rate'] = pd.to_numeric(df.iloc[:, -2], errors='coerce')
            df = df[['date', 'rate']]
            
            return df
            
        except Exception as e:
            print(f"Erreur €STR: {e}")
            return pd.DataFrame(columns=['date', 'rate'])
    
    def get_eur_swap_curve(self,
                           reference_date:str=None,
                           rate_level:float=3.5)-> pd.DataFrame:
        """
        Generate synthetic euro swap curve using Nelson-Siegel method with calibrated parameters
      
        Args:
            reference_date: reference date (default: today)
            rate_level: mean rate level (default:: 3.5%)
            
        Returns:
            DataFrame with columns: maturity_years, swap_rate, reference_date
        """

        if reference_date is None:
            reference_date=datetime.now().strftime("%Y-%m-%d")

        #Standard maturity for EU Market
        maturities=np.array([
            1/12, 3/12, 6/12,  # Short term
            1, 2, 3, 4, 5,      # Medium term
            7, 10, 12, 15,      # Long term
            20, 25, 30          # Very long term
        ])

        #Nelson Siegel parameters
        beta0= rate_level/100
        beta1= -0.005
        beta2= 0.008
        tau=2.5

        #NS formula
        rates=(
            beta0+
            beta1 * (1-np.exp(-maturities/tau))/(maturities/tau)+
            beta2 * ((1 - np.exp(-maturities/tau)) / (maturities/tau) - np.exp(-maturities/tau))
        )

        #Add realistic noise (bid ask spread)
        np.random.seed(42)
        noise=np.random.normal(0,0.0002, len(maturities))
        rates=rates+noise

        #constraints: no negativity
        rates=np.maximum(rates, 0.001)

        df=pd.DataFrame({
            "maturity_years": maturities,
            "swap_rate":rates*100, #in percentage
            "reference_date":reference_date
        })

        return df
    
    def generate_synthetic_eur_swaption_vols(self, 
                                             reference_date: str = None) -> pd.DataFrame:
        """
        Generate matrix of synthetic euro swaption volatility
        
        based on typical level of EU market
            
        Returns:
            DataFrame with: expiry, tenor, volatility
        """
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
        
        # Expiries et tenors standard
        expiries = [1, 2, 3, 5, 7, 10]
        tenors = [1, 2, 3, 5, 7, 10]
        
        data = []
        
        # base vol (ATM, in %)
        base_vol = 50.0  # 50% en vol Black normale
        
        for expiry in expiries:
            for tenor in tenors:
                # Vol decreases with expiry and tenor (smily effect)
                vol = base_vol * np.exp(-0.05 * expiry) * np.exp(-0.03 * tenor)
                
                # Adjustment for structure (hump)
                if expiry <= 3:
                    vol *= 1.1
                
                # noise
                noise = np.random.normal(0, 1.0)
                vol += noise
                
                data.append({
                    'expiry': expiry,
                    'tenor': tenor,
                    'volatility': vol,
                    'reference_date': reference_date
                })
        
        return pd.DataFrame(data)
    
    def generate_synthetic_eur_cap_vols(self, 
                                        reference_date: str = None) -> pd.DataFrame:
        """
        Returns:
            DataFrame with: maturity, cap_volatility
        """
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
        
        # Standard maturity for caps
        maturities = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30])
        
        # Volatilités ATM EUR (in %)
        base_vol = 45.0
        
        # volatility term structure
        vols = base_vol * np.exp(-0.04 * maturities) + 20.0
        
        # Noise adjustment
        np.random.seed(43)
        noise = np.random.normal(0, 1.5, len(maturities))
        vols += noise
        
        df = pd.DataFrame({
            'maturity': maturities,
            'cap_volatility': vols,
            'reference_date': reference_date
        })
        
        return df
    
    def create_complete_eur_dataset(self, 
                                    reference_date: str = None,
                                    use_synthetic: bool = True) -> Dict[str, pd.DataFrame]:
        """
        create complete dataset for HW calibration
        
        Args:
            reference_date: reference date
            use_synthetic: If True, use synthetic data 
            
        Returns:
            Dictionary composed of:
                - 'swap_curve'
                - 'swaption_vols'
                - 'cap_vols'
                - 'euribor'
        """
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Creating EU dataset for {reference_date}")
        
        dataset = {}
        
        if use_synthetic:
            print("Mode: Synthetic data (recommended for testing)")
            print("-" * 60)
            
            # Courbe de swap
            print("swap")
            dataset['swap_curve'] = self.get_eur_swap_curve(reference_date)
            
            # Volatilités swaptions
            print("swaption")
            dataset['swaption_vols'] = self.generate_synthetic_eur_swaption_vols(reference_date)
            
            # Volatilités caps
            print("cap_vols")
            dataset['cap_vols'] = self.generate_synthetic_eur_cap_vols(reference_date)
            
            # EURIBOR synthétique
            print("EURIBOR...")
            euribor_data = {
                'tenor': ['1M', '3M', '6M', '12M'],
                'rate': [3.85, 3.82, 3.75, 3.65],
                'reference_date': reference_date
            }
            dataset['euribor'] = pd.DataFrame(euribor_data)
            
        else:
            print("Mode: Actual data (pls have good internet connexion :>)")
            print("-" * 60)
            
            # Calcul des dates
            end_date = datetime.strptime(reference_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=365)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # EURIBOR réel
            try:
                print("EURIBOR from ECB incomiing")
                dataset['euribor'] = self.get_euribor_rates(start_str, end_str)
            except Exception as e:
                print(f" failed, maybe try synthetic data: {e}")
                dataset['euribor'] = pd.DataFrame({
                    'tenor': ['1M', '3M', '6M', '12M'],
                    'rate': [3.85, 3.82, 3.75, 3.65],
                    'reference_date': reference_date
                })
            
            # Pour swap curve et vols, use synthetic because it may varies across exact ECB series
            print("using synthetic data for swap/vols...")
            dataset['swap_curve'] = self.get_eur_swap_curve(reference_date)
            dataset['swaption_vols'] = self.generate_synthetic_eur_swaption_vols(reference_date)
            dataset['cap_vols'] = self.generate_synthetic_eur_cap_vols(reference_date)
        
        print("Complete!")

        
        # Affichage résumé
        self._print_dataset_summary(dataset)
        
        return dataset
    
    def _print_dataset_summary(self, dataset: Dict[str, pd.DataFrame]):
        """Dataset summary"""
        print("SUMMARY:")
        print("-" * 60)
        
        if 'swap_curve' in dataset:
            df = dataset['swap_curve']
            print(f"\n1. Swap curve EUR:")
            print(f"   - # of points: {len(df)}")
            print(f"   - Maturities: {df['maturity_years'].min():.2f}Y to {df['maturity_years'].max():.0f}Y")
            print(f"   - Rate: {df['swap_rate'].min():.2f}% - {df['swap_rate'].max():.2f}%")
        
        if 'swaption_vols' in dataset:
            df = dataset['swaption_vols']
            print(f"\n2. Swaptions Vols:")
            print(f"   - # of points: {len(df)}")
            print(f"   - Expiries: {df['expiry'].unique()}")
            print(f"   - Tenors: {df['tenor'].unique()}")
            print(f"   - Avg Vol: {df['volatility'].mean():.2f}%")
        
        if 'cap_vols' in dataset:
            df = dataset['cap_vols']
            print(f"\n3. Volatilités Caps:")
            print(f"   - # of points: {len(df)}")
            print(f"   - Maturities: {df['maturity'].min()}Y to {df['maturity'].max()}Y")
            print(f"   - Avg Vol: {df['cap_volatility'].mean():.2f}%")
        
        if 'euribor' in dataset:
            df = dataset['euribor']
            print(f"\n4. Tx EURIBOR:")
            if 'tenor' in df.columns:
                print(f"   - Tenors available: {df['tenor'].unique()}")
                for _, row in df.iterrows():
                    print(f"   - {row['tenor']}: {row['rate']:.2f}%")
    
    def save_dataset(self, dataset: Dict[str, pd.DataFrame], output_dir: str = './data'):
        """
        Save as csv files
        
        Args:
            dataset: dictionary of DataFrames
            output_dir: output repository
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in dataset.items():
            filename = f"{output_dir}/eur_{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
    
    def load_dataset(self, input_dir: str = './data') -> Dict[str, pd.DataFrame]:
        """
        load data of csv files
        
        Args:
            input_dir: repository of input
            
        Returns:
            Dictionary de DataFrames
        """
        dataset = {}
        files = {
            'swap_curve': 'eur_swap_curve.csv',
            'swaption_vols': 'eur_swaption_vols.csv',
            'cap_vols': 'eur_cap_vols.csv',
            'euribor': 'eur_euribor.csv'
        }
        
        for name, filename in files.items():
            filepath = f"{input_dir}/{filename}"
            try:
                dataset[name] = pd.read_csv(filepath)
                print(f"loaded: {filepath}")
            except FileNotFoundError:
                print(f"no found this bro: {filepath}")
        
        return dataset
    
    def get_historical_volatility(self, 
                                   rates_df: pd.DataFrame, 
                                   window: int = 30) -> pd.DataFrame:
        """
        
        Args:
            rates_df: DataFrame with 'date', 'rate'
            window:
            
        Returns:
            DataFrame
        """
        df = rates_df.copy()
        df = df.sort_values('date')
        
        # Calcul des rendements
        df['returns'] = df['rate'].pct_change()
        
        # Volatilité rolling
        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        return df
    
    def compute_correlation_matrix(self, 
                                    euribor_df: pd.DataFrame) -> pd.DataFrame:
        """
        correlation matrix for diff tenors (2F mainly)

        Args:
            euribor_df: DataFrame with  EURIBOR
            
        Returns:
            Corr matrix
        """
        # pivot for tenors as columns
        pivot_df = euribor_df.pivot(index='date', columns='tenor', values='rate')
        
        # variation
        returns = pivot_df.pct_change().dropna()
        
        # corr mat
        corr_matrix = returns.corr()
        
        return corr_matrix
    
def create_sample_eur_data_for_calibration(output_dir: str = './data') -> Dict[str, pd.DataFrame]:
    """
    main function to quick load complete dataset from ECB
    
    Args:
        output_dir:
        
    Returns:
        Dataset complete for calibration
    """
    collector = ECBDataCollector()
    
    # Créer dataset synthétique
    dataset = collector.create_complete_eur_dataset(
        reference_date='2024-06-30',
        use_synthetic=True
    )
    
    # Sauvegarder
    collector.save_dataset(dataset, output_dir)
    
    return dataset


# Test
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COLLECTEUR DE DONNÉES EUR - BANQUE CENTRALE EUROPÉENNE")
    print("="*70 + "\n")
    
    # Initialisation
    collector = ECBDataCollector()
    
    # Option 1: synthetic
    print("\n synthetic data complete")
    dataset = collector.create_complete_eur_dataset(
        reference_date='2024-06-30',
        use_synthetic=True
    )
    
    print(dataset['swap_curve'].head(10))
    print(dataset['swaption_vols'].head(10))
    print(dataset['cap_vols'].head(10))

    collector.save_dataset(dataset, output_dir='./data')
    
    # computation
    swap_df = dataset['swap_curve']
    print(f"\n swap:")
    print(f"  - 1Y: {swap_df[swap_df['maturity_years']==1]['swap_rate'].values[0]:.3f}%")
    print(f"  - 5Y: {swap_df[swap_df['maturity_years']==5]['swap_rate'].values[0]:.3f}%")
    print(f"  - 10Y: {swap_df[swap_df['maturity_years']==10]['swap_rate'].values[0]:.3f}%")
    print(f"  - 30Y: {swap_df[swap_df['maturity_years']==30]['swap_rate'].values[0]:.3f}%")
    
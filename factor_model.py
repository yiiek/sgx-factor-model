import numpy as np
import pandas as pd

class FactorModelEngine:
    def __init__(self, industry_dummies):
        self.dummies = industry_dummies.set_index('yf_ticker')
        
    def compute_daily_system_matrices(self, prices_df, metadata_df):
        """
        Calculates daily factor panels. Generates an endogenous market portfolio return 
        derived dynamically from the cap-weighted log-returns of the universe itself.
        """
        tickers = self.dummies.index.tolist()
        
        missing_tickers = [t for t in tickers if t not in prices_df.columns]
        if missing_tickers:
            raise ValueError(f"The following tickers are missing from downloaded price tables: {missing_tickers}")
            
        log_returns = np.log(prices_df[tickers] / prices_df[tickers].shift(1))
        
        # 1. Size Factor (Log Market Cap at Close)
        shares = metadata_df['sharesOutstanding'].reindex(tickers)
        daily_mcap = prices_df[tickers].multiply(shares, axis=1)
        daily_size = np.log(daily_mcap)
        
        # 2. Value Factor (Earnings-to-Price Ratio)
        eps = metadata_df['trailingEps'].reindex(tickers)
        daily_value = prices_df[tickers].rdiv(eps, axis=1)
        
        # 3. Daily Momentum Factor (T-252 to T-21 Rolling Log Return Sum)
        daily_momentum = log_returns.rolling(window=231).sum().shift(21)
        
        # 4. Daily Volatility Factor: 1Y Rolling Beta to the Universe Market Portfolio
        total_daily_mcap = daily_mcap.sum(axis=1)
        daily_weights = daily_mcap.div(total_daily_mcap, axis=0)
        
        # Compute the Endogenous Market Index Return: sum(Weight_i * Return_i)
        internal_bench_ret = (log_returns * daily_weights).sum(axis=1)
        
        daily_vol = pd.DataFrame(index=prices_df.index, columns=tickers)
        # Calibrated for a 1-year window (252 trading days) with a 6-month safety threshold (126 days)
        roll_bench_var = internal_bench_ret.rolling(window=252, min_periods=126).var()
        
        for t in tickers:
            roll_cov = log_returns[t].rolling(window=252, min_periods=126).cov(internal_bench_ret)
            daily_vol[t] = np.where(roll_bench_var > 0, roll_cov / roll_bench_var, 1.0)
            
        daily_size = daily_size.ffill().bfill()
        daily_value = daily_value.ffill().bfill()
        daily_momentum = daily_momentum.ffill().bfill()
        daily_vol = daily_vol.ffill().bfill()
        
        return daily_mcap, daily_size, daily_value, daily_momentum, daily_vol

    def build_exposure_matrix_for_date(self, date, daily_mcap, daily_size, daily_value, daily_momentum, daily_vol):
        tickers = self.dummies.index.tolist()
        mcap_t = daily_mcap.loc[date].reindex(tickers).fillna(1.0)
        
        std_df = pd.DataFrame({
            'Value': daily_value.loc[date].reindex(tickers),
            'Momentum': daily_momentum.loc[date].reindex(tickers),
            'Volatility': daily_vol.loc[date].reindex(tickers),
            'Size': daily_size.loc[date].reindex(tickers)
        }, index=tickers).fillna(0.0)
        
        std_df = std_df.fillna(std_df.median())
        
        total_mcap = mcap_t.sum()
        mcap_w = mcap_t / total_mcap if total_mcap > 0 else (1.0 / len(tickers))
        
        for col in std_df.columns:
            cw_mean = np.sum(std_df[col] * mcap_w)
            ew_std = std_df[col].std(ddof=1)
            std_df[col] = (std_df[col] - cw_mean) / (ew_std if ew_std != 0 else 1.0)
            
        matrix_df = self.dummies.copy()
        matrix_df['Market'] = 1.0  
        
        # Track updated SGX sector sorting hierarchy
        cols_order = ['Market', 'Banks', 'Real Estate', 'Others']
        return matrix_df[cols_order].join(std_df)

    def decompose_returns(self, prices_df, daily_mcap, daily_size, daily_value, daily_momentum, daily_vol, lookback_days=252):
        tickers = self.dummies.index.tolist()
        dates = prices_df.index[-lookback_days:]
        log_returns = np.log(prices_df[tickers] / prices_df[tickers].shift(1)).loc[dates]
        
        X_df = self.build_exposure_matrix_for_date(dates[-1], daily_mcap, daily_size, daily_value, daily_momentum, daily_vol)
        K = X_df.shape[1]
        industry_cols = ['Banks', 'Real Estate', 'Others']
        
        factor_returns_list = []
        residuals_list = []
        
        for date in dates:
            r_t = log_returns.loc[date].values
            X_df_t = self.build_exposure_matrix_for_date(date, daily_mcap, daily_size, daily_value, daily_momentum, daily_vol)
            X = X_df_t.values
            
            mcap_t = daily_mcap.loc[date].reindex(tickers).fillna(1.0).values
            weights = np.sqrt(mcap_t)
            W = np.diag(weights / np.sum(weights))
            
            total_mcap = np.sum(mcap_t)
            C = np.zeros(K)
            for i, col in enumerate(X_df_t.columns):
                if col in industry_cols:
                    sector_mcap = np.sum(mcap_t * X_df_t[col].values)
                    C[i] = sector_mcap / total_mcap if total_mcap > 0 else (1.0 / len(industry_cols))
            
            omega = X.T @ W @ X
            A_top = np.hstack([omega, C.reshape(-1, 1)])
            A_bot = np.hstack([C, [0.0]])
            A_block = np.vstack([A_top, A_bot])
            
            B_vector = np.concatenate([X.T @ W @ r_t, [0.0]])
            
            try:
                solution = np.linalg.solve(A_block, B_vector)
                f_t = solution[:K]
            except np.linalg.LinAlgError:
                f_t = np.linalg.pinv(X) @ r_t
                
            u_t = r_t - X @ f_t
            factor_returns_list.append(f_t)
            residuals_list.append(u_t)
            
        df_factor_returns = pd.DataFrame(factor_returns_list, index=dates, columns=X_df.columns)
        df_residuals = pd.DataFrame(residuals_list, index=dates, columns=tickers)
        latest_exposure_matrix = self.build_exposure_matrix_for_date(dates[-1], daily_mcap, daily_size, daily_value, daily_momentum, daily_vol)
        
        return df_factor_returns, df_residuals, latest_exposure_matrix

    def generate_portfolio_weights(self, exposure_matrix, residual_df, target_exposures):
        tickers = exposure_matrix.index.tolist()
        idio_var = residual_df.var(ddof=1).replace(0, 1e-4).fillna(1e-4)
        Delta_inv = np.linalg.inv(np.diag(idio_var.values))
        
        full_rank_exposure = exposure_matrix.drop(columns=['Others'])
        A = full_rank_exposure.values.T 
        b = np.array([target_exposures.get(f, 0.0) for f in full_rank_exposure.columns])
        
        try:
            A_Delta_AT_inv = np.linalg.inv(A @ Delta_inv @ A.T)
            w = Delta_inv @ A.T @ A_Delta_AT_inv @ b
            return pd.Series(w, index=tickers)
        except np.linalg.LinAlgError:
            return pd.Series(1.0 / len(tickers), index=tickers)
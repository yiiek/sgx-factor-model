import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

# Matplotlib embedding framework
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from csv_parser import CSVParser
from data_fetcher import DataFetcher
from factor_model import FactorModelEngine

class FactorModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SGX Barra-Style Factor Model System")
        self.root.geometry("1350x800")
        
        # Global Operational Data States
        self.stocks_df = None
        self.prices_df = None
        self.metadata_df = None
        self.exposure_matrix = None
        self.factor_returns = None
        self.residuals = None
        self.portfolio_df = None
        
        self.chart_canvas = None
        
        self.create_widgets()

    def create_widgets(self):
        # Top Action Panel
        top_frame = ttk.LabelFrame(self.root, text="Data Import & Model Constraints")
        top_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(top_frame, text="1. Import Stocks CSV", command=self.import_stocks).pack(side="left", padx=5, pady=5)
        ttk.Button(top_frame, text="2. Import Portfolio CSV", command=self.import_portfolio).pack(side="left", padx=5, pady=5)
        
        ttk.Label(top_frame, text="Rebalance Allowance (Std Dev):").pack(side="left", padx=(15, 2))
        self.allowance_var = tk.StringVar(value="0.1")
        ttk.Entry(top_frame, textvariable=self.allowance_var, width=6).pack(side="left", padx=5)
        
        ttk.Button(top_frame, text="3. Generate & Optimize Weights", command=self.generate_weights).pack(side="right", padx=5, pady=5)
        
        # Display Workspace Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # View Tabs 
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook) 
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Asset Decomposition")
        self.notebook.add(self.tab2, text="Daily Factor Risk Premiums")
        self.notebook.add(self.tab5, text="Factor Returns Chart") 
        self.notebook.add(self.tab3, text="Portfolio Exposures")
        self.notebook.add(self.tab4, text="Rebalancing Trade Panel")
        
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab5()
        self.setup_tab3()
        self.setup_tab4()

    def setup_tab1(self):
        self.tree_decomp = ttk.Treeview(
            self.tab1, 
            columns=("Ticker", "Name", "Latest Return", "Market", "Banks", "Real Estate", "Others", "Value", "Momentum", "Volatility", "Size", "Residual"), 
            show="headings"
        )
        self.tree_decomp.pack(fill="both", expand=True)
        
        for col in self.tree_decomp["columns"]:
            self.tree_decomp.heading(col, text=col)
            if col == "Name":
                self.tree_decomp.column(col, width=180, anchor="w") 
            elif col == "Ticker":
                self.tree_decomp.column(col, width=70, anchor="center")
            else:
                self.tree_decomp.column(col, width=90, anchor="center")

    def setup_tab2(self):
        self.tree_factors = ttk.Treeview(
            self.tab2, 
            columns=("Factor", "Daily (1D)", "Weekly (5D)", "Monthly (21D)", "60-Day (Qtr)"), 
            show="headings"
        )
        self.tree_factors.pack(fill="both", expand=True)
        
        for col in self.tree_factors["columns"]:
            self.tree_factors.heading(col, text=col)
            if col == "Factor":
                self.tree_factors.column(col, width=150, anchor="w")
            else:
                self.tree_factors.column(col, width=150, anchor="center")

    def setup_tab5(self):
        self.chart_label = ttk.Label(self.tab5, text="Import stock universe data to generate 60-day historical performance plots.", font=("Helvetica", 11, "italic"))
        self.chart_label.pack(expand=True, pady=20)

    def setup_tab3(self):
        main_p = ttk.Panedwindow(self.tab3, orient="horizontal")
        main_p.pack(fill="both", expand=True)
        
        self.tree_exp = ttk.Treeview(main_p, columns=("Factor", "Current Portfolio Exposure"), show="headings")
        for col in self.tree_exp["columns"]:
            self.tree_exp.heading(col, text=col)
            self.tree_exp.column(col, width=180, anchor="center")
            
        input_frame = ttk.LabelFrame(main_p, text="Portfolio Capital & Target Factor Tilts")
        
        ttk.Label(input_frame, text="Portfolio Gross Size (Capital $):", font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 2), sticky="w")
        self.portfolio_capital_var = tk.StringVar(value="100000") 
        ttk.Entry(input_frame, textvariable=self.portfolio_capital_var, width=15).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 15), sticky="w")
        
        self.target_entries = {}
        # RESTORED: 'Market' is fully integrated back into target parameters
        factors = ['Market', 'Banks', 'Real Estate', 'Others', 'Value', 'Momentum', 'Volatility', 'Size']
        
        for i, f in enumerate(factors):
            ttk.Label(input_frame, text=f"{f}:").grid(row=i+2, column=0, padx=10, pady=4, sticky="w")
            var = tk.StringVar(value="1.0" if f == 'Market' else "0.0")
            ent = ttk.Entry(input_frame, textvariable=var, width=10)
            ent.grid(row=i+2, column=1, padx=10, pady=4)
            self.target_entries[f] = var
            
        main_p.add(self.tree_exp, weight=3)
        main_p.add(input_frame, weight=1)

    def setup_tab4(self):
        self.tree_trades = ttk.Treeview(
            self.tab4, 
            columns=("Ticker", "Latest Close", "Current Shares", "Current Weight", "Target Weight", "Weight Diff", "Target Shares", "Trade Shares"), 
            show="headings"
        )
        self.tree_trades.pack(fill="both", expand=True)
        
        for col in self.tree_trades["columns"]:
            self.tree_trades.heading(col, text=col)
            if col == "Ticker":
                self.tree_trades.column(col, width=80, anchor="center")
            elif col in ["Latest Close", "Current Shares", "Target Shares", "Trade Shares"]:
                self.tree_trades.column(col, width=110, anchor="center")
            else:
                self.tree_trades.column(col, width=120, anchor="center")

    def import_stocks(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            self.stocks_df = CSVParser.parse_stocks_csv(file_path)
            tickers = self.stocks_df['yf_ticker'].tolist()
            
            fetcher = DataFetcher(tickers)
            self.prices_df = fetcher.fetch_historical_prices()
            self.metadata_df = fetcher.fetch_metadata()
            
            engine = FactorModelEngine(self.stocks_df)
            daily_mcap, daily_size, daily_value, daily_momentum, daily_vol = engine.compute_daily_system_matrices(
                self.prices_df, self.metadata_df
            )
            
            self.factor_returns, self.residuals, self.exposure_matrix = engine.decompose_returns(
                self.prices_df, daily_mcap, daily_size, daily_value, daily_momentum, daily_vol
            )
            
            self.populate_asset_decomposition()
            self.populate_factor_premiums()
            self.plot_factor_returns()
            
            messagebox.showinfo("Success", "System processing pipeline finalized for SGX universe.")
        except Exception as e:
            messagebox.showerror("Error", f"Processing halted: {str(e)}")

    def plot_factor_returns(self):
        if self.factor_returns is None or self.factor_returns.empty:
            return

        if hasattr(self, 'chart_label') and self.chart_label is not None:
            try:
                self.chart_label.destroy()
            except Exception:
                pass
            self.chart_label = None

        if self.chart_canvas is not None:
            try:
                self.chart_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.chart_canvas = None

        fig = Figure(figsize=(10, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        
        cumulative_factor_returns = self.factor_returns.cumsum()
        
        for factor in cumulative_factor_returns.columns:
            linewidth = 2.5 if factor == 'Market' else 1.2
            ax.plot(cumulative_factor_returns.index, cumulative_factor_returns[factor], label=factor, linewidth=linewidth)

        ax.set_title("60-Day Cumulative Factor Performance (Daily Log Return Path)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Timeline Date", fontsize=10)
        ax.set_ylabel("Cumulative Factor Log Return", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.tab5)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def import_portfolio(self):
        if self.exposure_matrix is None:
            messagebox.showerror("Dependency Error", "Please import a stocks layout CSV file first.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            self.portfolio_df = CSVParser.parse_portfolio_csv(file_path)
            self.populate_portfolio_exposures()
        except Exception as e:
            messagebox.showerror("Error", f"Failed parsing portfolio metrics: {str(e)}")

    def populate_asset_decomposition(self):
        for i in self.tree_decomp.get_children():
            self.tree_decomp.delete(i)
            
        tickers = self.exposure_matrix.index.tolist()
        log_returns = np.log(self.prices_df[tickers] / self.prices_df[tickers].shift(1))
        latest_ret = log_returns.iloc[-1]
        latest_f_ret = self.factor_returns.iloc[-1]
        
        for t in tickers:
            stk_name = self.metadata_df.loc[t, 'longName'] if 'longName' in self.metadata_df.columns else t
            ret_val = latest_ret[t] * 100
            row_exp = self.exposure_matrix.loc[t]
            contributions = (row_exp * latest_f_ret) * 100
            res_val = self.residuals.loc[self.residuals.index[-1], t] * 100
            
            self.tree_decomp.insert("", "end", values=(
                t, stk_name, f"{ret_val:+.3f}%",
                f"{contributions['Market']:+.3f}%", f"{contributions['Banks']:+.3f}%",
                f"{contributions['Real Estate']:+.3f}%", f"{contributions['Others']:+.3f}%",
                f"{contributions['Value']:+.3f}%", f"{contributions['Momentum']:+.3f}%",
                f"{contributions['Volatility']:+.3f}%", f"{contributions['Size']:+.3f}%",
                f"{res_val:+.3f}%"
            ))

    def populate_factor_premiums(self):
        for i in self.tree_factors.get_children():
            self.tree_factors.delete(i)
            
        if self.factor_returns is None or self.factor_returns.empty:
            return

        f_daily = self.factor_returns.iloc[-1] * 100
        f_weekly = self.factor_returns.iloc[-5:].sum() * 100
        f_monthly = self.factor_returns.iloc[-21:].sum() * 100
        f_60d = self.factor_returns.sum() * 100 

        for factor in self.factor_returns.columns:
            self.tree_factors.insert("", "end", values=(
                factor,
                f"{f_daily[factor]:+.4f}%",
                f"{f_weekly[factor]:+.4f}%",
                f"{f_monthly[factor]:+.4f}%",
                f"{f_60d[factor]:+.4f}%"
            ))

    def populate_portfolio_exposures(self):
        for i in self.tree_exp.get_children():
            self.tree_exp.delete(i)
            
        prices = self.prices_df[self.portfolio_df['yf_ticker']].iloc[-1]
        self.portfolio_df['Price'] = self.portfolio_df['yf_ticker'].map(prices)
        self.portfolio_df['MV'] = self.portfolio_df['Shares'] * self.portfolio_df['Price']
        total_mv = self.portfolio_df['MV'].sum()
        
        if total_mv == 0:
            return
            
        self.portfolio_df['Weight'] = self.portfolio_df['MV'] / total_mv
        
        w_series = self.portfolio_df.set_index('yf_ticker')['Weight'].reindex(self.exposure_matrix.index).fillna(0.0)
        port_exposures = self.exposure_matrix.T @ w_series
        
        for factor, val in port_exposures.items():
            self.tree_exp.insert("", "end", values=(factor, f"{val:.4f}"))

    def generate_weights(self):
        if self.exposure_matrix is None:
            messagebox.showerror("Error", "Missing base exposure records. Run stock calculations first.")
            return
            
        try:
            allowance = float(self.allowance_var.get())
            targets = {f: float(self.target_entries[f].get()) for f in self.target_entries}
            capital = float(self.portfolio_capital_var.get())
        except ValueError:
            messagebox.showerror("Validation Error", "Invalid numerical entry inputs inside workspace panels.")
            return
            
        tickers = self.exposure_matrix.index.tolist()
        
        if self.portfolio_df is not None:
            w_curr = self.portfolio_df.set_index('yf_ticker')['Weight'].reindex(tickers).fillna(0.0)
            shares_series = self.portfolio_df.set_index('yf_ticker')['Shares'].reindex(tickers).fillna(0.0)
        else:
            w_curr = pd.Series(0.0, index=tickers)
            shares_series = pd.Series(0.0, index=tickers)
            
        curr_exposures = self.exposure_matrix.T @ w_curr
        
        rebalance_triggered = False
        stylized_factors = ['Value', 'Momentum', 'Volatility', 'Size']
        
        for f in stylized_factors:
            if abs(targets[f] - curr_exposures.get(f, 0.0)) > allowance:
                rebalance_triggered = True
                break
                
        if rebalance_triggered or self.portfolio_df is None:
            engine = FactorModelEngine(self.stocks_df)
            w_target = engine.generate_portfolio_weights(self.exposure_matrix, self.residuals, targets)
        else:
            w_target = w_curr.copy()
            messagebox.showinfo("Status", "Current portfolios stay within risk bounds. Tracking trade actions bypassed.")
            
        for i in self.tree_trades.get_children():
            self.tree_trades.delete(i)
            
        for t in tickers:
            close_p = self.prices_df[t].iloc[-1]
            c_w = w_curr.get(t, 0.0)
            t_w = w_target.get(t, 0.0)
            diff_w = t_w - c_w
            
            curr_shares = int(round(shares_series.get(t, 0.0)))
            
            raw_target_shares = (t_w * capital) / close_p if close_p > 0 else 0
            
            # Formulates rounded 100-share SGX standard execution lot boundaries
            target_shares_rounded = int(round(raw_target_shares / 100.0)) * 100
            trade_shares = target_shares_rounded - curr_shares
            
            self.tree_trades.insert("", "end", values=(
                t, 
                f"${close_p:.2f}", 
                f"{curr_shares:,}",
                f"{c_w:.4%}", 
                f"{t_w:.4%}", 
                f"{diff_w:+.4%}",
                f"{target_shares_rounded:,}", 
                f"{trade_shares:+,}" if trade_shares != 0 else "0"
            ))
            
        self.notebook.select(self.tab4)
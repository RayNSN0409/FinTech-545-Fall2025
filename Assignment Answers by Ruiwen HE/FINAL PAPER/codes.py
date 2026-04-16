"""
Final paper workflow: SPY vs XLF tail-risk modeling with GJR-GARCH and EGARCH.

This script performs:
1) Data download and log-return construction
2) Descriptive statistics and return-series visualization
3) In-sample estimation (80%) for GJR-GARCH(1,1) and EGARCH(1,1)
4) Out-of-sample rolling one-step-ahead volatility forecasts (20%)
5) Dynamic VaR/ES at 95% and 99% under Normal innovations
6) Backtesting via VaR exception rates
7) Export of tables and publication-quality figures (PDF)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from arch import arch_model
from scipy.stats import norm

# Use a non-interactive backend to avoid Qt plugin issues on headless or misconfigured systems.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Global configuration
# -----------------------------
TICKERS = ["SPY", "XLF"]
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"

CONF_LEVELS = [0.95, 0.99]
TRAIN_RATIO = 0.80
DIST = "normal"

FIG_DPI = 300
ROLLING_REFIT_EVERY = 1


@dataclass(frozen=True)
class ModelSpec:
	name: str
	vol: str
	p: int = 1
	o: int = 1
	q: int = 1


MODEL_SPECS: Dict[str, ModelSpec] = {
	"GJR-GARCH": ModelSpec(name="GJR-GARCH", vol="GARCH", p=1, o=1, q=1),
	"EGARCH": ModelSpec(name="EGARCH", vol="EGARCH", p=1, o=1, q=1),
}


def ensure_dirs(*dirs: Path) -> None:
	for d in dirs:
		d.mkdir(parents=True, exist_ok=True)


def fetch_log_returns(ticker: str, start: str, end: str) -> pd.Series:
	"""Download adjusted close prices and return daily log returns in percent."""
	df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
	if df.empty:
		raise ValueError(f"No data downloaded for {ticker}.")

	if "Close" not in df.columns:
		raise ValueError(f"Expected 'Close' column is missing for {ticker}.")

	log_ret = np.log(df["Close"]).diff().dropna() * 100.0
	log_ret.name = ticker
	return log_ret


def descriptive_stats(series: pd.Series) -> pd.Series:
	"""Compute core moments for EDA and heavy-tail diagnostics."""
	return pd.Series(
		{
			"Mean(%)": series.mean(),
			"Std(%)": series.std(ddof=1),
			"Skewness": series.skew(),
			"Kurtosis": series.kurtosis() + 3.0,
			"Min(%)": series.min(),
			"Max(%)": series.max(),
			"Obs": series.shape[0],
		}
	)


def plot_return_series(returns_df: pd.DataFrame, out_file: Path) -> None:
	"""Plot SPY/XLF return series with stress-period highlights."""
	sns.set_theme(style="whitegrid", context="talk")
	fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

	stress_periods = [
		(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-15"), "COVID shock"),
		(pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-15"), "SVB crisis"),
	]

	for ax, ticker in zip(axes, returns_df.columns):
		ax.plot(returns_df.index, returns_df[ticker], color="#0b5fa5", linewidth=0.8)
		for start, end, label in stress_periods:
			ax.axvspan(start, end, alpha=0.15, color="#d62728")
			ax.text(
				start,
				ax.get_ylim()[1] * 0.80,
				label,
				color="#8b0000",
				fontsize=10,
			)
		ax.set_title(f"{ticker} Daily Log Returns (in %)" )
		ax.set_ylabel("Return (%)")

	axes[-1].xaxis.set_major_locator(mdates.YearLocator())
	axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
	axes[-1].set_xlabel("Date")

	plt.tight_layout()
	plt.savefig(out_file, dpi=FIG_DPI, bbox_inches="tight")
	plt.close(fig)


def build_arch_model(series: pd.Series, spec: ModelSpec):
	return arch_model(
		series,
		mean="Constant",
		vol=spec.vol,
		p=spec.p,
		o=spec.o,
		q=spec.q,
		dist=DIST,
		rescale=False,
	)


def fit_in_sample(series: pd.Series, spec: ModelSpec):
	model = build_arch_model(series, spec)
	return model.fit(disp="off")


def extract_estimation_table(ticker: str, model_name: str, fit_res) -> pd.DataFrame:
	params = fit_res.params
	pvals = fit_res.pvalues
	tvals = fit_res.tvalues

	rows = []
	for pname in params.index:
		rows.append(
			{
				"Ticker": ticker,
				"Model": model_name,
				"Parameter": pname,
				"Estimate": params[pname],
				"t-stat": tvals[pname],
				"p-value": pvals[pname],
				"AIC": fit_res.aic,
				"BIC": fit_res.bic,
				"LogLik": fit_res.loglikelihood,
			}
		)
	return pd.DataFrame(rows)


def rolling_vol_forecast(
	full_series: pd.Series,
	spec: ModelSpec,
	train_size: int,
	refit_every: int = 1,
) -> pd.Series:
	"""Fixed-window one-step-ahead rolling volatility forecast for out-of-sample."""
	if refit_every < 1:
		raise ValueError("refit_every must be >= 1")

	n_total = full_series.shape[0]
	n_test = n_total - train_size
	test_index = full_series.index[train_size:]
	sigmas = np.zeros(n_test, dtype=float)

	cached_fit = None
	for i in range(n_test):
		# Keep window length fixed at train_size for rolling forecasts.
		train_slice = full_series.iloc[i : i + train_size]

		if i % refit_every == 0 or cached_fit is None:
			model = build_arch_model(train_slice, spec)
			cached_fit = model.fit(disp="off")

		fcst = cached_fit.forecast(horizon=1, reindex=False)
		variance_1d = float(fcst.variance.values[-1, 0])
		sigmas[i] = np.sqrt(max(variance_1d, 0.0))

	return pd.Series(sigmas, index=test_index, name=f"sigma_{spec.name}")


def var_es_normal(
	sigma: pd.Series,
	conf_levels: Iterable[float],
	mu: float = 0.0,
) -> pd.DataFrame:
	"""Compute left-tail VaR and ES under Normal assumptions."""
	out = pd.DataFrame(index=sigma.index)

	for cl in conf_levels:
		alpha_tail = 1.0 - cl
		z = norm.ppf(alpha_tail)
		var = mu + sigma * z
		es = mu - sigma * (norm.pdf(z) / alpha_tail)
		label = int(cl * 100)
		out[f"VaR_{label}"] = var
		out[f"ES_{label}"] = es

	return out


def backtest_var(actual_returns: pd.Series, var_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	n = actual_returns.shape[0]

	for col in [c for c in var_df.columns if c.startswith("VaR_")]:
		level = int(col.split("_")[1])
		expected_rate = 1.0 - (level / 100.0)
		exceptions = (actual_returns < var_df[col]).sum()
		rate = exceptions / n
		rows.append(
			{
				"VaR Level": f"{level}%",
				"Expected Exception Rate": expected_rate,
				"Observed Exceptions": int(exceptions),
				"Observed Exception Rate": rate,
				"Abs Error": abs(rate - expected_rate),
			}
		)

	return pd.DataFrame(rows)


def backtest_es(actual_returns: pd.Series, var_es_df: pd.DataFrame) -> pd.DataFrame:
	"""Evaluate ES quality on VaR exceedance days via realized tail-loss comparison."""
	rows = []

	for col in [c for c in var_es_df.columns if c.startswith("VaR_")]:
		level = int(col.split("_")[1])
		es_col = f"ES_{level}"
		mask = actual_returns < var_es_df[col]
		n_exc = int(mask.sum())

		if n_exc > 0:
			realized_tail_mean = float(actual_returns[mask].mean())
			predicted_es_mean = float(var_es_df.loc[mask, es_col].mean())
			es_gap = realized_tail_mean - predicted_es_mean
			abs_gap = abs(es_gap)
		else:
			realized_tail_mean = np.nan
			predicted_es_mean = np.nan
			es_gap = np.nan
			abs_gap = np.nan

		rows.append(
			{
				"ES Level": f"{level}%",
				"Tail Obs (VaR breaches)": n_exc,
				"Realized Tail Mean": realized_tail_mean,
				"Predicted ES Mean": predicted_es_mean,
				"ES Gap (Realized-Predicted)": es_gap,
				"Abs ES Gap": abs_gap,
			}
		)

	return pd.DataFrame(rows)


def plot_var_overlay(
	ticker: str,
	model_name: str,
	actual_returns: pd.Series,
	var_df: pd.DataFrame,
	out_file: Path,
) -> None:
	sns.set_theme(style="whitegrid", context="talk")
	fig, ax = plt.subplots(figsize=(14, 6))

	ax.scatter(
		actual_returns.index,
		actual_returns.values,
		s=12,
		alpha=0.6,
		color="#1f77b4",
		label="Actual Return",
	)
	ax.plot(var_df.index, var_df["VaR_95"], color="#ff7f0e", linewidth=1.4, label="95% VaR")
	ax.plot(var_df.index, var_df["VaR_99"], color="#d62728", linewidth=1.4, label="99% VaR")

	ax.set_title(f"{ticker} Out-of-Sample Returns vs Dynamic VaR ({model_name})")
	ax.set_xlabel("Date")
	ax.set_ylabel("Return (%)")
	ax.legend(loc="best")

	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.savefig(out_file, dpi=FIG_DPI, bbox_inches="tight")
	plt.close(fig)


def plot_var_es_overlay(
	ticker: str,
	model_name: str,
	actual_returns: pd.Series,
	var_es_df: pd.DataFrame,
	out_file: Path,
) -> None:
	sns.set_theme(style="whitegrid", context="talk")
	fig, ax = plt.subplots(figsize=(14, 6))

	ax.scatter(
		actual_returns.index,
		actual_returns.values,
		s=10,
		alpha=0.55,
		color="#1f77b4",
		label="Actual Return",
	)
	ax.plot(var_es_df.index, var_es_df["VaR_95"], color="#ff7f0e", linewidth=1.4, label="95% VaR")
	ax.plot(var_es_df.index, var_es_df["VaR_99"], color="#d62728", linewidth=1.4, label="99% VaR")
	ax.plot(var_es_df.index, var_es_df["ES_95"], color="#2ca02c", linewidth=1.2, linestyle="--", label="95% ES")
	ax.plot(var_es_df.index, var_es_df["ES_99"], color="#9467bd", linewidth=1.2, linestyle="--", label="99% ES")

	ax.set_title(f"{ticker} Out-of-Sample Returns vs Dynamic VaR and ES ({model_name})")
	ax.set_xlabel("Date")
	ax.set_ylabel("Return (%)")
	ax.legend(loc="best", ncol=2)

	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.savefig(out_file, dpi=FIG_DPI, bbox_inches="tight")
	plt.close(fig)


def run_pipeline(base_dir: Path) -> None:
	out_tables = base_dir / "outputs" / "tables"
	out_figs = base_dir / "outputs" / "figures"
	out_data = base_dir / "outputs" / "data"
	ensure_dirs(out_tables, out_figs, out_data)

	# 1) Data fetch
	returns = {}
	for tk in TICKERS:
		returns[tk] = fetch_log_returns(tk, START_DATE, END_DATE)

	returns_df = pd.concat(returns.values(), axis=1).dropna()
	returns_df.columns = TICKERS
	returns_df.to_csv(out_data / "daily_log_returns_percent.csv", index=True)

	# 2) EDA
	desc_table = pd.concat(
		[descriptive_stats(returns_df[c]).rename(c) for c in returns_df.columns],
		axis=1,
	).T
	desc_table.to_csv(out_tables / "descriptive_statistics.csv", index=True)
	plot_return_series(returns_df, out_figs / "figure1_returns_with_stress_periods.pdf")

	# 3) In-sample fit and information criteria
	train_size = int(TRAIN_RATIO * len(returns_df))
	estimation_tables = []
	ic_rows = []

	all_var_es = {}
	all_backtests = []
	all_es_backtests = []

	for ticker in TICKERS:
		full_series = returns_df[ticker].copy()
		in_sample = full_series.iloc[:train_size]
		out_sample = full_series.iloc[train_size:]

		for model_name, spec in MODEL_SPECS.items():
			fit_res = fit_in_sample(in_sample, spec)
			estimation_tables.append(extract_estimation_table(ticker, model_name, fit_res))

			ic_rows.append(
				{
					"Ticker": ticker,
					"Model": model_name,
					"AIC": fit_res.aic,
					"BIC": fit_res.bic,
					"LogLik": fit_res.loglikelihood,
				}
			)

			# 4) Rolling one-step-ahead volatility forecast
			sigma_hat = rolling_vol_forecast(
				full_series=full_series,
				spec=spec,
				train_size=train_size,
				refit_every=ROLLING_REFIT_EVERY,
			)

			# 5) VaR / ES computation
			var_es_df = var_es_normal(sigma_hat, CONF_LEVELS, mu=0.0)
			merged = pd.concat([out_sample.rename("ActualReturn"), sigma_hat, var_es_df], axis=1)

			key = (ticker, model_name)
			all_var_es[key] = merged
			merged.to_csv(out_data / f"var_es_{ticker}_{model_name}.csv", index=True)

			# 6) Backtest (exceptions)
			bt = backtest_var(out_sample, var_es_df)
			bt.insert(0, "Model", model_name)
			bt.insert(0, "Ticker", ticker)
			all_backtests.append(bt)

			es_bt = backtest_es(out_sample, var_es_df)
			es_bt.insert(0, "Model", model_name)
			es_bt.insert(0, "Ticker", ticker)
			all_es_backtests.append(es_bt)

			# 7) VaR overlay chart
			plot_var_overlay(
				ticker=ticker,
				model_name=model_name,
				actual_returns=out_sample,
				var_df=var_es_df,
				out_file=out_figs / f"figure2_var_overlay_{ticker}_{model_name}.pdf",
			)

			plot_var_es_overlay(
				ticker=ticker,
				model_name=model_name,
				actual_returns=out_sample,
				var_es_df=var_es_df,
				out_file=out_figs / f"figure3_var_es_overlay_{ticker}_{model_name}.pdf",
			)

	# Export all tables
	pd.concat(estimation_tables, axis=0, ignore_index=True).to_csv(
		out_tables / "model_parameter_estimates.csv", index=False
	)
	pd.DataFrame(ic_rows).to_csv(out_tables / "information_criteria.csv", index=False)
	pd.concat(all_backtests, axis=0, ignore_index=True).to_csv(
		out_tables / "var_backtest_summary.csv", index=False
	)
	pd.concat(all_es_backtests, axis=0, ignore_index=True).to_csv(
		out_tables / "es_backtest_summary.csv", index=False
	)


if __name__ == "__main__":
	base = Path(__file__).resolve().parent
	run_pipeline(base)
	print("Pipeline completed. Outputs are saved under ./outputs/")

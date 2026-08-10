# KOSPI200 CCI(9) + DMI(14) + Parabolic SAR Backtest

- Universe: current KOSPI200 constituents, frozen for the whole period
- Period: 2023-08-10 ~ 2026-08-10
- Entry: CCI(9) crosses above 0 AND PSAR bullish AND +DI(14) > -DI(14)
- Exit: CCI < 0 OR PSAR bearish OR +DI <= -DI
- Signal at close, execution at next trading day's open
- GitHub Actions runs both gross and fee-adjusted tests
- Fee-adjusted workflow uses 0.10% per side

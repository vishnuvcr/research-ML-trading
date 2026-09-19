# Data Contract

## Market bars
Required: timestamp, symbol, exchange, open, high, low, close, volume, source, source_version, retrieved_at, timezone, adjusted_flag.

## CPR features
Required: reference_session, signal_session, P, BC, TC, cpr_width, cpr_width_pct, position features, breakout/rejection flags, regime features, feature_version.

## Trade ledger
Required: trade_id, decision_timestamp, fill_timestamp, symbol, side, quantity, intended_price, fill_price, gross_pnl, slippage_cost, brokerage, exchange_cost, STT, SEBI_fee, GST, stamp_duty, DP_other_cost, funding_cost, total_cost, net_pnl, exit_reason, execution_model_version.

## Provenance
Every canonical dataset needs source/provider, retrieval timestamp, date range, row count, checksum, transformation commit and protocol version.

## Cache
Reuse cache when requested dataset/checksum matches. Refresh only for a source/data correction or explicit version change.

## Licensing
Do not store or redistribute provider data when prohibited by its licence. When raw storage is not permitted, retain metadata, checksum, retrieval parameters and permitted derived data.

## Validation gates
Phase 01 fails on unresolved duplicate timestamps, invalid OHLC, chronology violations, unknown timezone, unexpected gaps or schema mismatch.

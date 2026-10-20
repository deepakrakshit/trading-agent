# Market data and cost-source audit

Checked on **19 October 2026**. No market-data subscription was purchased, no exchange market pages were scraped, and no authenticated provider call was made.

## Sources inspected

| Source | Official reference inspected | Observed capability/access | Decision |
| --- | --- | --- | --- |
| Angel One SmartAPI | `https://smartapi.angelone.in/faq`, `https://smartapi.angelone.in/docs` | Both returned “You are not authorize to access this page.” Current free-access claim, entitlements, official limits and terms could not be reverified. | Do not assert free/live access; mock-test only. |
| Angel One public SDK | `https://github.com/angel-one/smartapi-python` | Public README/source document historical candles, quote REST, and WebSocket V2. Session examples require account authentication; SDK also exposes real order calls and can log headers. | Do **not** install/import the broad trading SDK. Implement a narrow read-only transport with only candle and quote paths. |
| Zerodha historical | `https://kite.trade/docs/connect/v3/historical/` | Authenticated timestamp/OHLCV candles, including `5minute`; instrument tokens are explicit. | Supported conceptually through authorised CSV import; no account was used. |
| Zerodha pricing | `https://zerodha.com/products/api/` | Page lists Connect with real-time WebSocket/historical candles at **₹500/month**. Free Personal tier lists order/portfolio functionality, not free historical streaming. | No purchase or free-data assumption. |
| Upstox V3 | `https://upstox.com/developer/api-documentation/v3/get-market-data-feed/` | Bearer-authenticated, Protobuf WebSocket feed; normal limit two connections; mode-dependent subscriptions; heartbeat is not a fresh trade. | Not installed/integrated; current account entitlements/costs not established. |
| Local fixtures | `workflow.generate_synthetic` | Deterministic artificial weekday sessions; seed 2026; no real NSE observations. | Fully offline engineering/training demo, always labelled **SYNTHETIC**. |

The public Angel One SDK's HEAD was verified as `7f10dad8baad16c4c17d658818cfd574a90cbf53`. The audited source files were its `README.md` and `SmartApi/smartConnect.py`; no downloaded script was executed. The official source confirms these routes on `https://apiconnect.angelone.in`:

- `/rest/secure/angelbroking/historical/v1/getCandleData`
- `/rest/secure/angelbroking/market/v1/quote`

These are the **only** allowed HTTP routes in the application's provider. Order placement/modification/cancellation, login, holdings, order updates and arbitrary URLs are absent. Redirects cannot forward credentials. Transport failures, 429/backoff and auth expiry are bounded, and errors do not include raw request headers/bodies.

## Authentication, cost, reliability and licensing limitations

- A user must independently confirm an eligible provider account, data entitlement, terms, storage/use rights and any charges. Public API documentation is not a licence to redistribute data.
- `.env.example` contains placeholders only. The CLI reads locally exported session/API-key variables; it neither logs in nor stores provider sessions. Do not paste secrets into prompts or CLI arguments.
- SmartAPI's default **one request/second** is a conservative *local cap*, not a claimed official rate limit. Current official limits must be confirmed locally. Retries respect bounded `Retry-After`; auth failures stop rather than loop.
- Reliability is tested with scripted disconnects/timeouts/429s/malformed/stale responses. **Actual provider uptime, response conventions, permissions and latency remain unverified.**
- `quote` returns **READ_ONLY_POLL** observations with provider and receipt timestamps; it is not called verified LIVE. A fresh heartbeat with an old/missing last-trade timestamp is rejected.
- WebSocket functionality was researched in the official SDK, but live entitlement/access, a verified live NSE feed, and deployment-grade operational reconnection were not validated. The repository now has a provider-independent completed-bar replay/polling boundary with bounded retry, but it must not be described as a working live NSE feed.
- The quote adapter additionally rejects an optional provider `tradingSymbol`
  that disagrees with the caller's explicit symbol/token mapping. This is
  scripted-response evidence only; it does not verify a real instrument master.

## Canonical data handling

The application stores immutable CSV/Parquet data with a separate manifest: source, provenance, symbols, actual inclusive bar-start range, capture time, row count, schema, calendar and SHA-256. Source metadata is allowlisted; secrets/request headers cannot be passed as manifest fields. Parquet stores canonical Decimal/text values losslessly, then numerical features are generated separately.

Bar timestamps are **BAR_START** in aware UTC; exchange sessions use **Asia/Kolkata**. Five-minute features become available at bar start + five minutes. The environment logs `decision_available_at`, next-open `fill_timestamp`, and `reward_available_at` explicitly. At an adjacent bar boundary this assumes a zero-latency decision with adverse spread/slippage; real execution latency is not reconstructed from OHLCV.

Validation rejects malformed OHLCV, NaN/infinity/nonpositive prices, duplicate/out-of-order rows, incomplete/forming candles, future/unavailable records, wrong symbols, missing scheduled intraday candles, and unreviewed 20% discontinuities/outlier ranges. It never sorts away a defect, forward-fills a tradable price, or silently switches provenance. Missing/zero volume cannot invent fill liquidity. At an open, participation uses **prior completed volume**, not the future completed volume of the fill candle.

The generic CSV/snapshot boundary enforces the same five-minute `BAR_START`
contract: CSV `complete` values are strict booleans, capture time must be at
least one interval after every imported bar start, and unknown columns are
rejected. Snapshot manifest hashes are pinned into new experiment and
paper-account receipts; reloads validate against the original snapshot capture
time rather than the current wall clock.

The default calendar is explicitly a **weekday NSE-style schedule**, not an authoritative holiday calendar. Synthetic sessions can include dates such as 26 January that are not real NSE trading sessions. Supply verified holidays/exceptional sessions for real data. Corporate actions, adjusted-vs-raw prices and point-in-time symbol/universe histories require source-specific work; large discontinuities fail closed rather than fabricate an adjustment.

## Versioned cost assumptions

`https://zerodha.com/charges` was read during this session. It lists equity intraday brokerage 0.03% capped at ₹20/order, sell-side STT 0.025%, NSE transaction charge 0.00307%, SEBI ₹10/crore, 18% GST on the applicable service charges, buy stamp duty 0.003%, and equity IPFT ₹0.01/crore plus GST.

`configs/default.json` uses these proportions as an **illustrative intraday research schedule**, not an Angel One bill or a contractual current tariff. Its version is `illustrative-intraday-2026-10-19-v1`; the date is the audit/configuration date, not an invented statutory effective date. Every rate remains configurable/versioned.

The simulator uses integer-paise monetary rounding, a ₹0.05 illustrative price tick, 4 bps full spread (2 bps per side) plus 3 bps adverse slippage per side, and separately itemized fees. Slippage is in the fill price once, never charged twice. Exact contract-note aggregation, statutory rounding, delivery/DP fees, corporate-action cash flows, and broker-specific auto-square-off charges are not reproduced. Default operation precommits a 15:25 open exit; a rejected exit leaves a marked position and a failure flag. Such a carry is **not** qualified as accurately delivery-costed performance.

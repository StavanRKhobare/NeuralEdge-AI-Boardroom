const formatMoney = (n) =>
    n >= 1e6 ? `$${(n / 1e6).toFixed(2)}M`
        : n >= 1e3 ? `$${(n / 1e3).toFixed(1)}K`
            : `$${Math.abs(n).toFixed(0)}`

const formatPct = (v) => `${(v * 100).toFixed(0)}%`
const fmtDelta = (key, d) => {
    if (key === 'revenue' || key === 'burn_rate') return formatMoney(Math.abs(d))
    if (key === 'runway_months') return `${Math.abs(d).toFixed(1)}mo`
    if (key === 'profitability_score') return Math.abs(d).toFixed(1)
    return formatPct(Math.abs(d))
}

// ASCII progress bar — no chart.js, pure terminal
function AsciiBar({ value, max = 1, width = 8 }) {
    const filled = Math.round((value / max) * width)
    const empty = width - filled
    return (
        <span style={{ color: 'var(--muted)', letterSpacing: 0 }}>
            [<span style={{ color: 'var(--primary)', textShadow: 'var(--glow-sm)' }}>
                {'█'.repeat(Math.max(0, filled))}
            </span>
            {'░'.repeat(Math.max(0, empty))}]
        </span>
    )
}

const TILES = [
    { key: 'profitability_score', label: 'SCORE',    fmt: (v) => v.toFixed(1),            max: 100 },
    { key: 'revenue',             label: 'REVENUE',  fmt: formatMoney,                     max: null },
    { key: 'burn_rate',           label: 'BURN',     fmt: formatMoney,                     max: null },
    { key: 'runway_months',       label: 'RUNWAY',   fmt: (v) => `${v.toFixed(1)}mo`,      max: 24 },
    { key: 'product_readiness',   label: 'PRODUCT',  fmt: formatPct,                       max: 1 },
    { key: 'market_share',        label: 'MARKET',   fmt: formatPct,                       max: 1 },
    { key: 'team_morale',         label: 'MORALE',   fmt: formatPct,                       max: 1 },
    { key: 'investor_confidence', label: 'INVEST',   fmt: formatPct,                       max: 1 },
    { key: 'regulatory_risk',     label: 'REG_RSK',  fmt: formatPct,                       max: 1 },
]

function scoreTile(key, val) {
    if (key === 'regulatory_risk') return val > 0.65 ? 'bad' : val > 0.35 ? 'warn' : 'good'
    if (key === 'runway_months')   return val > 12   ? 'good' : val > 6   ? 'warn' : 'bad'
    if (key === 'profitability_score') return val >= 60 ? 'good' : val >= 35 ? 'warn' : 'bad'
    if (key === 'burn_rate') return ''
    return val > 0.65 ? 'good' : val > 0.35 ? 'warn' : 'bad'
}

export default function MetricsPanel({ state, prevState }) {
    if (!state) return null

    return (
        <div className="metrics-strip">
            {TILES.map(({ key, label, fmt, max }) => {
                const val   = state[key] ?? 0
                const prev  = prevState?.[key]
                const delta = prev !== undefined ? val - prev : null
                const cls   = scoreTile(key, val)
                const barVal = max ? Math.min(val, max) : null

                return (
                    <div key={key} className="metric-tile">
                        <div className="m-icon-label">
                            <span className="m-label">{label}</span>
                        </div>
                        <div className="m-value-row">
                            <span className={`m-value ${cls}`}>{fmt(val)}</span>
                            {delta !== null && Math.abs(delta) > 0.001 && (
                                <span className={`m-delta ${delta > 0 ? 'pos' : 'neg'}`}>
                                    {delta > 0 ? '+' : '−'}{fmtDelta(key, delta)}
                                </span>
                            )}
                        </div>
                        {barVal !== null && (
                            <div style={{ marginTop: '0.15rem' }}>
                                <AsciiBar value={barVal} max={max} width={6} />
                            </div>
                        )}
                    </div>
                )
            })}
        </div>
    )
}

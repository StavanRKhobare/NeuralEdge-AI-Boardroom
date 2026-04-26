export default function RewardTrace({ trace = [], cumReward = 0, lastReward = null }) {
    const bounds = trace.length
        ? trace.reduce(
            (acc, v) => ({ min: Math.min(acc.min, v), max: Math.max(acc.max, v) }),
            { min: 0, max: 0 },
        )
        : { min: 0, max: 0 }

    const lo = Math.min(bounds.min, -0.5)
    const hi = Math.max(bounds.max, 1.0)
    const span = Math.max(0.1, hi - lo)
    const zeroFrac = (0 - lo) / span

    const cls =
        lastReward === null ? '' :
            lastReward > 0.5 ? 'good' :
                lastReward > 0 ? 'warn' :
                    'bad'

    return (
        <div className="card" aria-label="Per-step reward trace">
            <div className="section-label">Reward Trace · Episode</div>
            <div className="card-body" style={{ padding: '0.5rem 0.75rem 0.65rem' }}>
                <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    fontSize: '0.6rem', color: 'var(--text-muted)', marginBottom: '0.35rem',
                }}>
                    <span>step={trace.length}/10</span>
                    <span>cum={cumReward.toFixed(2)}</span>
                </div>

                <div className="reward-bars">
                    {Array.from({ length: 10 }).map((_, i) => {
                        const v = trace[i]
                        const has = typeof v === 'number'
                        const upFrac = has && v > 0 ? Math.min(1, v / Math.max(hi, 0.01)) : 0
                        const downFrac = has && v < 0 ? Math.min(1, -v / Math.max(-lo, 0.01)) : 0
                        const upPct = `${(upFrac * 100 * (1 - zeroFrac)).toFixed(0)}%`
                        const downPct = `${(downFrac * 100 * zeroFrac).toFixed(0)}%`
                        return (
                            <div
                                key={i}
                                className={`reward-bar-slot ${has ? (v >= 0 ? 'pos' : 'neg') : 'empty'}`}
                                title={has ? `R${i + 1}: reward ${v >= 0 ? '+' : ''}${v.toFixed(3)}` : `R${i + 1}: pending`}
                            >
                                <div className="reward-bar-up"   style={{ height: upPct,   bottom: `${zeroFrac * 100}%` }} />
                                <div className="reward-bar-down" style={{ height: downPct, top:    `${(1 - zeroFrac) * 100}%` }} />
                                <div className="reward-bar-zero" style={{ bottom: `${zeroFrac * 100}%` }} />
                                <div className="reward-bar-label">{i + 1}</div>
                            </div>
                        )
                    })}
                </div>

                <div style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                    marginTop: '0.4rem', fontSize: '0.65rem',
                }}>
                    <span style={{ color: 'var(--text-muted)' }}>last_step</span>
                    <span className={`m-value ${cls}`}>
                        {lastReward === null ? '—' : `${lastReward >= 0 ? '+' : ''}${Number(lastReward).toFixed(3)}`}
                    </span>
                </div>
            </div>
        </div>
    )
}

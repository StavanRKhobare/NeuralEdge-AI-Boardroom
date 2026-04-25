const formatMoney = (n) =>
    n >= 1e6 ? `$${(n / 1e6).toFixed(2)}M`
        : n >= 1e3 ? `$${(n / 1e3).toFixed(1)}K`
            : `$${n?.toFixed(0) ?? 0}`

const OUTCOME_MAP = {
    ipo:              { ascii: '[IPO]',    title: 'IPO_SUCCESS',       cls: 'ipo' },
    acquisition:      { ascii: '[ACQ]',    title: 'ACQUIRED',          cls: 'acquisition' },
    runway_exhausted: { ascii: '[DEAD]',   title: 'BANKRUPTCY',        cls: 'bankruptcy' },
    finished_10:      { ascii: '[DONE]',   title: 'EPISODE_COMPLETE',  cls: 'default' },
}

const DIVIDER = '================================================'

export default function EndScreen({ obs, onReplay }) {
    if (!obs) return null
    const { state }   = obs
    const reason      = state?.done_reason ?? 'finished_10'
    const { ascii, title, cls } = OUTCOME_MAP[reason] ?? OUTCOME_MAP['finished_10']
    const history     = state?.history ?? []
    const roundsWon   = history.filter(h => h.agent_won_vote).length

    return (
        <div className="end-overlay">
            <div className="end-modal">
                {/* ASCII banner */}
                <div className="end-icon" style={{ fontSize: '1.5rem', fontWeight: 700, letterSpacing: '0.1em' }}>
                    {ascii}
                </div>
                <div className="end-title">{title}</div>
                <span className={`end-reason ${cls}`}>{reason.replace(/_/g, '_')}</span>

                <div style={{ color: 'var(--muted)', fontSize: '0.6rem', marginBottom: '0.875rem', letterSpacing: '0.05em' }}>
                    {DIVIDER}
                </div>

                <div className="end-stats">
                    <div className="end-stat">
                        <div className="es-label">PROFIT_SCORE</div>
                        <div className="es-value" style={{ color: 'var(--primary)', textShadow: 'var(--glow)' }}>
                            {(state?.profitability_score ?? 0).toFixed(1)}
                        </div>
                    </div>
                    <div className="end-stat">
                        <div className="es-label">REVENUE</div>
                        <div className="es-value">{formatMoney(state?.revenue ?? 0)}</div>
                    </div>
                    <div className="end-stat">
                        <div className="es-label">RUNWAY</div>
                        <div className="es-value">{(state?.runway_months ?? 0).toFixed(1)}mo</div>
                    </div>
                    <div className="end-stat">
                        <div className="es-label">ROUNDS_WON</div>
                        <div className="es-value" style={{ color: 'var(--primary)', textShadow: 'var(--glow)' }}>
                            {roundsWon}/{history.length}
                        </div>
                    </div>
                    <div className="end-stat">
                        <div className="es-label">MORALE</div>
                        <div className="es-value">{Math.round((state?.team_morale ?? 0) * 100)}%</div>
                    </div>
                    <div className="end-stat">
                        <div className="es-label">REG_RISK</div>
                        <div className="es-value">{Math.round((state?.regulatory_risk ?? 0) * 100)}%</div>
                    </div>
                </div>

                {history.length > 0 && (
                    <div style={{ marginBottom: '1rem', maxHeight: '180px', overflowY: 'auto' }}>
                        <div style={{
                            fontSize: '0.58rem', textTransform: 'uppercase', letterSpacing: '0.1em',
                            color: 'var(--text-secondary)', marginBottom: '0.4rem'
                        }}>
                            // round_log
                        </div>
                        {history.map((h) => (
                            <div key={h.round} style={{
                                display: 'flex', justifyContent: 'space-between',
                                fontSize: '0.65rem', color: 'var(--text-secondary)',
                                padding: '0.2rem 0',
                                borderBottom: '1px solid var(--border-dim)'
                            }}>
                                <span style={{ color: 'var(--secondary)', textShadow: 'var(--amber-glow)', minWidth: '28px' }}>
                                    R{String(h.round).padStart(2,'0')}
                                </span>
                                <span style={{
                                    flex: 1, marginLeft: '0.6rem',
                                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                    textTransform: 'uppercase', letterSpacing: '0.03em',
                                    fontSize: '0.6rem'
                                }}>
                                    {(h.event_title ?? '').split('—').slice(-1)[0]?.trim()}
                                </span>
                                <span style={{
                                    marginLeft: '0.5rem', flexShrink: 0, fontSize: '0.6rem',
                                    color: h.agent_won_vote ? 'var(--primary)' : 'var(--error)',
                                    textShadow: h.agent_won_vote ? 'var(--glow-sm)' : 'none'
                                }}>
                                    {h.agent_won_vote ? '[OK]' : '[X]'} {(h.agent_decision ?? '').replace(/_/g, '_')}
                                </span>
                            </div>
                        ))}
                    </div>
                )}

                <button className="replay-btn" onClick={onReplay}>
                    ↺ RUN_NEW_EPISODE
                </button>
            </div>
        </div>
    )
}

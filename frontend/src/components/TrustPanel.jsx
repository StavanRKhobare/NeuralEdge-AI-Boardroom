const ROLES = ['CTO', 'CFO', 'Investor Rep', 'Independent']
const KEY_MAP = { 'CTO': 'cto', 'CFO': 'cfo', 'Investor Rep': 'inv', 'Independent': 'ind' }

// ASCII block bar
function AsciiTrustBar({ pct }) {
    const total  = 12
    const filled = Math.round((pct / 100) * total)
    return (
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', letterSpacing: 0 }}>
            {'█'.repeat(filled)}{'░'.repeat(total - filled)}
        </span>
    )
}

export default function TrustPanel({ trust, prevTrust }) {
    if (!trust) {
        return (
            <div className="card">
                <div className="section-label">Board Trust</div>
                <div className="card-body" style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    // awaiting first round...
                </div>
            </div>
        )
    }

    return (
        <div className="card">
            <div className="section-label">Board Trust</div>
            <div className="trust-list">
                {ROLES.map((role) => {
                    const val   = trust[role] ?? 0.5
                    const prev  = prevTrust?.[role]
                    const delta = prev !== undefined ? val - prev : null
                    const cls   = KEY_MAP[role]
                    const pct   = Math.round(val * 100)
                    return (
                        <div key={role} className="trust-row">
                            <span className="trust-role">{role}</span>
                            <div className="trust-track">
                                <div
                                    className={`trust-fill ${cls}`}
                                    style={{ width: `${pct}%` }}
                                />
                            </div>
                            <span className="trust-pct">{pct}%</span>
                            {delta !== null && Math.abs(delta) > 0.001 ? (
                                <span className={`trust-delta ${delta > 0 ? 'pos' : 'neg'}`}>
                                    {delta > 0 ? '▲' : '▼'}
                                </span>
                            ) : (
                                <span className="trust-delta" />
                            )}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

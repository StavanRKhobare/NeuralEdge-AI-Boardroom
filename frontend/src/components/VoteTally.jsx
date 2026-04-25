// ASCII block bar for vote weight visualisation
function VoteBar({ pct, isWinner }) {
    const total  = 14
    const filled = Math.round((pct / 100) * total)
    const char   = isWinner ? '█' : '▒'
    const color  = isWinner ? 'var(--primary)' : 'var(--text-muted)'
    const shadow = isWinner ? 'var(--glow-sm)' : 'none'
    return (
        <span style={{ color, textShadow: shadow, letterSpacing: 0, fontFamily: 'var(--font-mono)', fontSize: '0.62rem' }}>
            {char.repeat(Math.max(0, filled))}{'░'.repeat(Math.max(0, total - filled))}
        </span>
    )
}

export default function VoteTally({ info }) {
    const tally = info?.winning_vote_tally
    if (!tally) return null

    const winner = Object.entries(tally).reduce((a, b) => (b[1] > a[1] ? b : a), ['', 0])[0]
    const max    = Math.max(...Object.values(tally), 0.01)

    return (
        <div className="card">
            <div className="section-label">Vote Tally</div>
            <div className="vote-tally">
                {Object.entries(tally).map(([opt, val]) => {
                    const pct      = Math.round((val / max) * 100)
                    const isWinner = opt === winner
                    return (
                        <div key={opt} className="tally-row">
                            <span className="tally-label">{opt.replace(/_/g, '_')}</span>
                            <VoteBar pct={pct} isWinner={isWinner} />
                            <span className="tally-value">{val.toFixed(1)}</span>
                        </div>
                    )
                })}
            </div>
            <div className="tally-winner-row">
                winner: <span style={{ color: 'var(--primary)', textShadow: 'var(--glow)' }}>
                    {winner.replace(/_/g, '_')}
                </span>
            </div>
        </div>
    )
}

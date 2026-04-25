// NPC agenda keyword hints shown on cards (top 4 per role)
const AGENDA_HINTS = {
    'CTO':           ['engineering', 'architecture', 'team morale', 'reliability'],
    'CFO':           ['burn rate', 'runway', 'fiduciary', 'cost discipline'],
    'Investor Rep':  ['growth', 'market share', 'IPO', 'bold moves'],
    'Independent':   ['reputation', 'ethics', 'long-term', 'governance'],
}

const ROLE_CLS = {
    'CTO': 'cto', 'CFO': 'cfo', 'Investor Rep': 'inv', 'Independent': 'ind',
}

const ROLE_INITIALS = {
    'CTO': 'CT', 'CFO': 'CF', 'Investor Rep': 'IN', 'Independent': 'ID',
}

function NPCCard({ npc }) {
    const { role, statement, vote, confidence } = npc
    const cls  = ROLE_CLS[role]  ?? 'ind'
    const pct  = Math.round(confidence * 100)
    const hints = AGENDA_HINTS[role] ?? []

    return (
        <div className={`npc-card ${cls}`}>
            <div className="npc-header">
                <div className="npc-avatar-role">
                    <div className={`npc-avatar ${cls}`}>{ROLE_INITIALS[role] ?? role[0]}</div>
                    <span className={`npc-role ${cls}`}>{role.toUpperCase()}</span>
                </div>
                <span className="npc-vote-chip" title={`Votes: ${vote}`}>
                    →{vote.replace(/_/g, '_')}
                </span>
            </div>

            <p className="npc-statement">{statement}</p>

            <div className="npc-conf-row">
                <span className="conf-label">CONF</span>
                <div className="conf-track">
                    <div className="conf-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="conf-pct">{pct}%</span>
            </div>

            <div className="npc-agenda-tags">
                {hints.map((h) => (
                    <span key={h} className="agenda-tag">#{h}</span>
                ))}
            </div>
        </div>
    )
}

export default function NPCGrid({ npcStatements }) {
    if (!npcStatements?.length) {
        return (
            <div className="card">
                <div className="section-label">Board Statements</div>
                <div className="card-body" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
                    // awaiting board response...
                </div>
            </div>
        )
    }

    return (
        <div className="card">
            <div className="section-label">Board Statements</div>
            <div className="npc-grid">
                {npcStatements.map((npc) => (
                    <NPCCard key={npc.role} npc={npc} />
                ))}
            </div>
        </div>
    )
}

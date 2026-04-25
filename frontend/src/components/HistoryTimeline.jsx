export default function HistoryTimeline({ history }) {
    return (
        <div className="card">
            <div className="section-label">Decision History</div>
            {!history?.length ? (
                <div className="history-empty">// no rounds completed yet.</div>
            ) : (
                <div className="history-list">
                    {history.map((entry) => {
                        const aiWon     = entry.agent_won_vote
                        const reward    = entry.reward ?? ((entry.score_after ?? 0) - 0)
                        const rewardNum = typeof reward === 'number' ? reward : 0
                        return (
                            <div key={entry.round} className="history-item">
                                <span className="h-round">R{String(entry.round).padStart(2,'0')}</span>
                                <div className="h-info">
                                    <div className="h-event">
                                        {(entry.event_title ?? '').split('—').slice(-1)[0]?.trim() ?? entry.event_title}
                                    </div>
                                    <div className="h-picks">
                                        <span className="h-ai-pick">
                                            &gt;{(entry.agent_decision ?? '').replace(/_/g, '_')}
                                        </span>
                                        {!aiWon && (
                                            <>
                                                <span style={{ color: 'var(--muted)' }}>→</span>
                                                <span className="h-win-pick">
                                                    {(entry.winning_decision ?? '').replace(/_/g, '_')}
                                                </span>
                                                <span className="h-mismatch">[X]</span>
                                            </>
                                        )}
                                        {aiWon && (
                                            <span style={{ color: 'var(--primary)', fontSize: '0.55rem', textShadow: 'var(--glow-sm)' }}>
                                                &nbsp;[OK]
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <span className={`h-reward ${rewardNum >= 0 ? 'pos' : 'neg'}`}>
                                    {rewardNum >= 0 ? '+' : ''}{rewardNum.toFixed(2)}
                                </span>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

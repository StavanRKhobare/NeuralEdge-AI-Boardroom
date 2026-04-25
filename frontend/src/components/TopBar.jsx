import { useEffect, useState } from 'react'
import { apiHealth } from '../services/api.js'

const ASCII_LOGO = `
 _  _ ____ _  _ ____ ____ _    ____ ___  ____ ____
 |\ | |___ |  | |__/ |__| |    |___ |  \ | __ |___
 | \| |___ |__| |  \ |  | |___ |___ |__/ |__] |___
`

export default function TopBar({ obs, round }) {
    const [online, setOnline] = useState(false)
    const [tick, setTick] = useState(true)

    useEffect(() => {
        const check = async () => setOnline(await apiHealth())
        check()
        const id = setInterval(check, 15_000)
        return () => clearInterval(id)
    }, [])

    // blinking colon in clock-style indicator
    useEffect(() => {
        const t = setInterval(() => setTick(v => !v), 500)
        return () => clearInterval(t)
    }, [])

    const score = obs?.state?.profitability_score ?? null
    const scoreClass = score === null ? '' : score >= 60 ? 'good' : score >= 35 ? 'warn' : 'bad'

    return (
        <div className="topbar">
            <div className="topbar-brand">
                {/* compact single-line ASCII header */}
                <div className="brand-name">NeuralEdge</div>
                <div style={{ width: '1px', height: '18px', background: 'var(--border)', margin: '0 0.35rem' }} />
                <div className="brand-ceo">CEO: Sarah Chen&nbsp;|&nbsp;AI Agent</div>
            </div>

            <div className="topbar-center">
                <div className="round-badge">
                    RND {obs ? `${String(round).padStart(2,'0')} / 10` : '--/10'}
                </div>
                {score !== null && (
                    <div className="score-display">
                        <div className="score-label">PROFIT_SCORE</div>
                        <div className={`score-value ${scoreClass}`}>{score.toFixed(1)}</div>
                    </div>
                )}
            </div>

            <div className="topbar-right">
                <div className="health-indicator">
                    <div className={`health-dot ${online ? 'online' : 'offline'}`} />
                    {online ? '[OK] BACKEND' : '[ERR] OFFLINE'}
                </div>
            </div>
        </div>
    )
}

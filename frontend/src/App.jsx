import { useEffect, useState } from 'react'
import { useGameStore } from './hooks/useGameStore.js'
import { useAgentLoop, greedyPick, buildPitch } from './hooks/useAgentLoop.js'

import TopBar from './components/TopBar.jsx'
import PlaybackControls from './components/PlaybackControls.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'
import TrustPanel from './components/TrustPanel.jsx'
import EventBanner from './components/EventBanner.jsx'
import NPCGrid from './components/NPCGrid.jsx'
import AgentDecision from './components/AgentDecision.jsx'
import VoteTally from './components/VoteTally.jsx'
import HistoryTimeline from './components/HistoryTimeline.jsx'
import EndScreen from './components/EndScreen.jsx'

export default function App() {
    const { state, resetGame, stepGame, setSpeed, setPaused } = useGameStore()
    const { obs, prevObs, done, loading, error, lastReward, lastInfo, speed, paused } = state

    const [toast, setToast] = useState(null)

    // Show error toast
    useEffect(() => {
        if (error) {
            setToast(error)
            const t = setTimeout(() => setToast(null), 5000)
            return () => clearTimeout(t)
        }
    }, [error])

    // Boot
    useEffect(() => { resetGame(42) }, [resetGame])

    // Wire agent loop
    useAgentLoop(state, stepGame)

    const handleRun = () => setPaused(false)
    const handlePause = () => setPaused(true)
    const handleReset = () => { resetGame(Math.floor(Math.random() * 9999)) }
    const handleReplay = () => { resetGame(Math.floor(Math.random() * 9999)) }

    const handleStep = async () => {
        if (!obs || loading || done) return
        const decision = greedyPick(obs)
        const pitch = buildPitch(obs, decision)
        if (decision) await stepGame(decision, pitch)
    }

    const round = obs?.round ?? 0
    const curState = obs?.state
    const prevState = prevObs?.state

    return (
        <div className="app-shell">
            <TopBar obs={obs} round={round} />
            <PlaybackControls
                paused={paused}
                loading={loading}
                done={done}
                obs={obs}
                speed={speed}
                onRun={handleRun}
                onPause={handlePause}
                onStep={handleStep}
                onReset={handleReset}
                onSpeedChange={setSpeed}
            />

            {/* Metrics strip at top — always visible */}
            {curState && (
                <MetricsPanel state={curState} prevState={prevState} />
            )}

            <div className="main-grid">
                {/* Left — Trust + History */}
                <div className="col-left">
                    <TrustPanel trust={curState?.trust} prevTrust={prevState?.trust} />
                    <HistoryTimeline history={curState?.history} />
                </div>

                {/* Centre — Event + NPCs + Agent Decision */}
                <div className="col-center">
                    <EventBanner event={obs?.event} round={round} />
                    <NPCGrid npcStatements={obs?.npc_statements} />
                    <AgentDecision obs={obs} loading={loading} lastInfo={lastInfo} />
                </div>

                {/* Right — Vote Tally */}
                <div className="col-right">
                    {lastInfo?.winning_vote_tally && <VoteTally info={lastInfo} />}
                    {!lastInfo && (
                        <div className="card">
                            <div className="section-label">Vote Tally</div>
                            <div className="card-body" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1.25rem 1rem' }}>
                                // vote tally appears after first decision.
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {done && obs && <EndScreen obs={obs} onReplay={handleReplay} />}

            {toast && (
                <div className="toast">
                    ⚠ {toast}
                </div>
            )}
        </div>
    )
}

export default function PlaybackControls({
    paused,
    loading,
    done,
    obs,
    speed,
    onRun,
    onPause,
    onStep,
    onReset,
    onSpeedChange,
}) {
    const canStep = !loading && !done && !!obs
    const statusText = loading ? 'PROCESSING...'
        : done ? 'EPISODE_DONE'
            : paused ? 'PAUSED'
                : 'RUNNING'

    const statusDot = loading ? '' : done ? '' : paused ? 'paused' : 'running'

    return (
        <div className="playback-bar">
            {paused && !done ? (
                <button className="pb-btn primary" onClick={onRun} disabled={loading || !obs}>
                    ▶ RUN_AGENT
                </button>
            ) : (
                <button className="pb-btn" onClick={onPause} disabled={loading || done}>
                    ⏸ PAUSE
                </button>
            )}

            <button className="pb-btn" onClick={onStep} disabled={!canStep}>
                ⏭ STEP
            </button>

            <div className="pb-divider" />

            <button className="pb-btn" onClick={onReset} disabled={loading}>
                ↺ RESET
            </button>

            <div className="pb-divider" />

            <div className="speed-control">
                <span>SPEED</span>
                <input
                    type="range"
                    min={0.5}
                    max={4}
                    step={0.25}
                    value={speed}
                    onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
                />
                <span className="speed-label">{speed.toFixed(2)}x</span>
            </div>

            <div className="pb-status">
                <div className={`status-dot ${statusDot}`} />
                {statusText}
            </div>
        </div>
    )
}

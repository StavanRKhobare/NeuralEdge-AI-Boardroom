export default function EventBanner({ event, round }) {
    if (!event) {
        return (
            <div className="event-banner">
                <div className="event-tag">BOARD_CRISIS</div>
                <div className="event-title">Awaiting scenario...</div>
                <div className="event-desc">$ run_agent --start  # click RUN_AGENT to begin</div>
            </div>
        )
    }

    const [titlePart, ...rest] = event.split('\n')
    const desc = rest.join(' ').replace(/^Description:\s*/i, '').trim() || event

    return (
        <div className="event-banner">
            <div className="event-tag">
                RND_{String(round).padStart(2,'0')} / BOARD_CRISIS
            </div>
            <div className="event-title">{titlePart.toUpperCase()}</div>
            {desc && desc !== titlePart && (
                <div className="event-desc">{desc}</div>
            )}
        </div>
    )
}

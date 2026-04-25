// ── NeuralEdge API Service ────────────────────────────────────────
// Talks to the STATEFUL /game/* endpoints (not the stateless /reset + /step).
// Base URL: VITE_API_BASE env var OR http://localhost:8000

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function _post(path, body) {
    const res = await fetch(`${BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(`${path} → ${res.status}: ${text}`)
    }
    return res.json()
}

/**
 * POST /game/reset — start a stateful episode (persists server-side).
 * @param {number} seed
 * @returns {Promise<{ observation, reward, done, info }>}
 */
export async function apiReset(seed = 42) {
    return _post('/game/reset', { seed })
}

/**
 * POST /game/step — advance the persistent episode by one round.
 * @param {string} decision
 * @param {string} coalitionPitch
 * @returns {Promise<{ observation, reward, done, info }>}
 */
export async function apiStep(decision, coalitionPitch = '') {
    return _post('/game/step', {
        decision,
        coalition_pitch: coalitionPitch,
    })
}

/**
 * GET /health — backend liveness check.
 * @returns {Promise<boolean>}
 */
export async function apiHealth() {
    try {
        const res = await fetch(`${BASE}/health`)
        const data = await res.json()
        return data.status === 'healthy'
    } catch {
        return false
    }
}
/**
 * POST /qwen/decide — ask the backend Qwen/Ollama proxy for a decision.
 * Falls back to greedy on the server side if Ollama is unavailable.
 * @param {object} obs  The current BoardSimObservation from state
 * @returns {Promise<{ decision: string, coalition_pitch: string, source: string }>}
 */
export async function apiQwenDecide(obs) {
    return _post('/qwen/decide', {
        state:          obs.state ?? {},
        event:          obs.event ?? '',
        options:        obs.options ?? [],
        npc_statements: obs.npc_statements ?? [],
        round:          obs.round ?? 1,
    })
}

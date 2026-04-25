// ── Game Store (useReducer) ───────────────────────────────────────
// Manages all game state consumed by the UI.

import { useReducer, useCallback } from 'react'
import { apiReset, apiStep } from '../services/api.js'

// ── Initial State ─────────────────────────────────────────────────
const initialState = {
    obs: null,         // current BoardSimObservation
    prevObs: null,     // previous obs for delta rendering
    done: false,
    loading: false,
    error: null,
    lastReward: null,
    lastInfo: null,
    speed: 1.5,        // playback speed multiplier
    paused: true,      // start paused, user clicks Run
    seed: 42,
}

// ── Reducer ───────────────────────────────────────────────────────
function reducer(state, action) {
    switch (action.type) {
        case 'RESET_START':
            return { ...initialState, speed: state.speed, loading: true, paused: true }

        case 'RESET_SUCCESS':
            return {
                ...state,
                loading: false,
                obs: action.payload.observation,
                prevObs: null,
                done: action.payload.done,
                lastReward: null,
                lastInfo: action.payload.info,
                error: null,
            }

        case 'STEP_START':
            return { ...state, loading: true }

        case 'STEP_SUCCESS':
            return {
                ...state,
                loading: false,
                prevObs: state.obs,
                obs: action.payload.observation,
                done: action.payload.done,
                lastReward: action.payload.reward,
                lastInfo: action.payload.info,
                error: null,
            }

        case 'SET_SPEED':
            return { ...state, speed: action.payload }

        case 'TOGGLE_PAUSE':
            return { ...state, paused: !state.paused }

        case 'SET_PAUSED':
            return { ...state, paused: action.payload }

        case 'ERROR':
            return { ...state, loading: false, error: action.payload, paused: true }

        default:
            return state
    }
}

// ── Hook ──────────────────────────────────────────────────────────
export function useGameStore() {
    const [state, dispatch] = useReducer(reducer, initialState)

    const resetGame = useCallback(async (seed = 42) => {
        dispatch({ type: 'RESET_START' })
        try {
            const data = await apiReset(seed)
            dispatch({ type: 'RESET_SUCCESS', payload: data })
        } catch (err) {
            dispatch({ type: 'ERROR', payload: err.message })
        }
    }, [])

    const stepGame = useCallback(async (decision, pitch = '') => {
        dispatch({ type: 'STEP_START' })
        try {
            const data = await apiStep(decision, pitch)
            dispatch({ type: 'STEP_SUCCESS', payload: data })
            if (data.done) dispatch({ type: 'SET_PAUSED', payload: true })
            return data
        } catch (err) {
            dispatch({ type: 'ERROR', payload: err.message })
            return null
        }
    }, [])

    const setSpeed = useCallback((v) => dispatch({ type: 'SET_SPEED', payload: v }), [])
    const togglePause = useCallback(() => dispatch({ type: 'TOGGLE_PAUSE' }), [])
    const setPaused = useCallback((v) => dispatch({ type: 'SET_PAUSED', payload: v }), [])

    return { state, resetGame, stepGame, setSpeed, togglePause, setPaused }
}

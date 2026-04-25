# BoardSim — Frontend API Specification

## Overview
The frontend communicates with the backend via REST/HTTP or WebSocket endpoints. The backend is a FastAPI server running at a configurable base URL (default: `http://localhost:8000` for local dev, or `https://<USER>-board-sim-env.hf.space` for production).

**Key Principle**: Frontend and backend are fully decoupled. The frontend only needs to know these endpoints; it does not import any backend code.

---

## 1. REST Endpoints

### `POST /reset`
**Purpose**: Start a new game episode.

**Request Body**:
```json
{
  "seed": 42,
  "episode_id": "optional-uuid-string"
}
```

**Response** (200 OK):
```json
{
  "observation": {
    "state": {
      "round": 1,
      "revenue": 2000000.0,
      "burn_rate": 1200000.0,
      "runway_months": 14.0,
      "product_readiness": 0.45,
      "market_share": 0.08,
      "team_morale": 0.70,
      "investor_confidence": 0.65,
      "regulatory_risk": 0.20,
      "profitability_score": 0.0,
      "trust": {
        "CTO": 0.5,
        "CFO": 0.5,
        "Investor Rep": 0.5,
        "Independent": 0.5
      },
      "trust_history": [
        {
          "round": 0,
          "CTO": 0.5,
          "CFO": 0.5,
          "Investor Rep": 0.5,
          "Independent": 0.5
        }
      ],
      "history": [],
      "done_reason": null,
      "winning_decision": null
    },
    "event": "New Competitor Entry — A larger competitor enters your core market with aggressive pricing and threatens your customer base.",
    "options": [
      "cut_prices",
      "double_down_on_quality",
      "form_strategic_partnership"
    ],
    "npc_statements": [
      {
        "role": "CTO",
        "statement": "From an operational standpoint, the trade-offs here are clear. I'm voting double_down_on_quality.",
        "vote": "double_down_on_quality",
        "confidence": 0.78
      },
      {
        "role": "CFO",
        "statement": "From a fiduciary standpoint, only one of these is defensible. I'm voting cut_prices.",
        "vote": "cut_prices",
        "confidence": 0.66
      },
      {
        "role": "Investor Rep",
        "statement": "We were not funded to play it safe. I'm voting form_strategic_partnership.",
        "vote": "form_strategic_partnership",
        "confidence": 0.70
      },
      {
        "role": "Independent",
        "statement": "Long-term reputation outlasts any single quarter. I'm voting double_down_on_quality.",
        "vote": "double_down_on_quality",
        "confidence": 0.59
      }
    ],
    "round": 1
  },
  "done": false,
  "info": {
    "episode_id": "uuid-string",
    "seed": 42
  }
}
```

---

### `POST /step`
**Purpose**: Submit the agent's decision for the current round.

**Request Body**:
```json
{
  "action": {
    "decision": "double_down_on_quality",
    "coalition_pitch": "Investing in product quality protects long-term reputation and reduces operational risk while preserving margin discipline."
  }
}
```

**Response** (200 OK):
```json
{
  "observation": {
    "state": {
      "round": 2,
      "revenue": 2000000.0,
      "burn_rate": 900000.0,
      "runway_months": 18.5,
      "product_readiness": 0.45,
      "market_share": 0.08,
      "team_morale": 0.65,
      "investor_confidence": 0.60,
      "regulatory_risk": 0.20,
      "profitability_score": 12.34,
      "trust": {
        "CTO": 0.65,
        "CFO": 0.70,
        "Investor Rep": 0.40,
        "Independent": 0.55
      },
      "trust_history": [
        {
          "round": 0,
          "CTO": 0.5,
          "CFO": 0.5,
          "Investor Rep": 0.5,
          "Independent": 0.5
        },
        {
          "round": 1,
          "CTO": 0.65,
          "CFO": 0.70,
          "Investor Rep": 0.40,
          "Independent": 0.55
        }
      ],
      "history": [
        {
          "round": 1,
          "event_title": "Round 1 — Series-B runway crunch",
          "agent_decision": "cut_costs",
          "winning_decision": "cut_costs",
          "reward": 1.25,
          "profitability_before": 0.0,
          "profitability_after": 12.34
        }
      ],
      "done_reason": null,
      "winning_decision": "cut_costs"
    },
    "event": "Round 2 — Enterprise contract w/ source-code escrow\nDescription: A Fortune 500 enterprise wants to sign a $5M contract but demands source code escrow.",
    "options": [
      "accept_deal",
      "negotiate_terms",
      "reject_deal"
    ],
    "npc_statements": [
      {
        "role": "CTO",
        "statement": "...",
        "vote": "...",
        "confidence": 0.XX
      }
    ],
    "round": 2
  },
  "reward": 1.25,
  "done": false,
  "info": {
    "round": 2,
    "winning_decision": "cut_costs",
    "winning_vote_tally": {
      "cut_costs": 4.2,
      "raise_capital": 1.3,
      "reduce_scope": 0.5
    },
    "pitch_scores": {
      "CTO": 0.0,
      "CFO": 0.0,
      "Investor Rep": 0.0,
      "Independent": 0.0
    }
  }
}
```

---

### `GET /health`
**Purpose**: Health check. Confirms backend is running.

**Response** (200 OK):
```json
{
  "status": "healthy"
}
```

---

### `GET /docs`
**Purpose**: Auto-generated Swagger/OpenAPI documentation. Use for development reference.

**Location**: `http://localhost:8000/docs` (or on HF Space at `/docs`)

---

## 2. WebSocket Streaming (Optional, Advanced)

If you want real-time streaming during training or multi-agent play:

### `WebSocket /ws`
**Purpose**: Bi-directional message streaming (not required for single-agent frontend).

Connection example:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message); // e.g., { "type": "step", "observation": {...} }
};
```

*(Details omitted if not used for initial frontend.)*

---

## 3. Data Models Reference

### `BoardSimObservation` (returned by `/reset` and `/step`)
```javascript
{
  state: {
    round: number,                        // 1-indexed: 1..10
    revenue: number,                      // in dollars
    burn_rate: number,                    // monthly spend in dollars
    runway_months: number,                // months until bankruptcy
    product_readiness: float (0..1),
    market_share: float (0..1),
    team_morale: float (0..1),
    investor_confidence: float (0..1),
    regulatory_risk: float (0..1),
    profitability_score: number,
    trust: {                              // per NPC, 0..1
      "CTO": 0.5,
      "CFO": 0.5,
      "Investor Rep": 0.5,
      "Independent": 0.5
    },
    trust_history: Array,                 // per-round trust snapshots
    history: Array,                       // past decisions & outcomes
    done_reason: string | null,           // e.g., "bankruptcy", "acquisition", "ipo", null
    winning_decision: string | null
  },
  event: string,                          // event title + description
  options: [string, string, string],      // 3 valid decision strings for this round
  npc_statements: [
    {
      role: "CTO" | "CFO" | "Investor Rep" | "Independent",
      statement: string,
      vote: string (one of options),
      confidence: float (0..1)
    },
    // ... one per NPC role (4 total)
  ],
  round: number
}
```

### `BoardSimAction` (sent to `/step`)
```javascript
{
  decision: string,                    // must be one of observation.options
  coalition_pitch: string | null       // optional persuasion attempt (unused in v1)
}
```

---

## 4. Error Responses

### 422 Unprocessable Entity
Invalid action format or decision not in options.

**Response**:
```json
{
  "detail": [
    {
      "loc": ["body", "action", "decision"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```

### 400 Bad Request
Malformed JSON or missing required fields.

---

## 5. Frontend Integration Checklist

- [ ] **Initialize**: On app load, call `POST /reset` to get initial observation.
- [ ] **Display State**: Render `observation.state` as metrics (revenue, runway, morale, trust, etc.).
- [ ] **Display Event**: Show `observation.event` (crisis title + description).
- [ ] **Display NPCs**: Render 4 NPC cards with their `statement`, `vote`, and `confidence`.
- [ ] **Render Decision Options**: Display 3 buttons (or cards) for each string in `observation.options`.
- [ ] **Handle User Click**: On decision click, POST `/step` with the selected `decision`.
- [ ] **Update UI**: Parse response observation and repeat from "Display State".
- [ ] **Terminal State**: If `done` is true, show final metrics and `done_reason` (e.g., "Bankruptcy", "IPO").
- [ ] **Optional Coalition Pitch**: Text input for `coalition_pitch` (future extension; safe to leave blank for v1).

---

## 6. Backend Base URL Configuration

For local development:
```
http://localhost:8000
```

For HF Space deployment (after `openenv push`):
```
https://<your-hf-username>-board-sim-env.hf.space
```

**Frontend environment variable** (optional):
```
REACT_APP_API_BASE_URL=http://localhost:8000
// or
REACT_APP_API_BASE_URL=https://<your-hf-username>-board-sim-env.hf.space
```

---

## 7. Example Frontend Workflow

```javascript
// 1. Reset
const resetRes = await fetch(`${API_BASE}/reset`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ seed: 42 })
});
const { observation, done, info } = await resetRes.json();

// 2. Render observation
displayState(observation.state);
displayNPCStatements(observation.npc_statements);
displayDecisionButtons(observation.options);

// 3. User clicks decision
const decision = "cut_costs"; // from button click
const stepRes = await fetch(`${API_BASE}/step`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    action: { decision, coalition_pitch: "" }
  })
});
const { observation: nextObs, reward, done: nextDone } = await stepRes.json();

// 4. Repeat or show results
if (nextDone) {
  displayEndgameScreen(nextObs.state, nextObs.state.done_reason);
} else {
  displayState(nextObs.state);
  // ... repeat
}
```

---

## 8. No Backend Imports in Frontend

✅ **OK**: `fetch("http://localhost:8000/reset")`
❌ **NOT OK**: `import { BoardSimEnvironment } from "backend"`

The frontend is a standalone web app. All communication is via HTTP/WebSocket.


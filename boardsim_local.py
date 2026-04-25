"""
boardsim_local.py
=================
Self-contained local GRPO training script for the BoardSim environment
(generic, organization-agnostic boardroom simulation). No HuggingFace tokens,
no WandB, no Docker, no HF Spaces required.

Note: this is the *local dev script*. The canonical OpenEnv environment that
runs on the HF Space and that the notebook trains against lives in
`envs/board_sim_env/server/board_sim_env_environment.py` and has the full
10-event timeline, sentence-transformer pitch scorer, and CEO weight 2.5x.

Requirements (pip install before running):
    pip install torch transformers trl>=0.12 datasets accelerate matplotlib numpy peft

Run as a regular Python script or paste cells into a Jupyter notebook.
"""

# ── 0. Installs (uncomment if running in a fresh notebook) ───────────────────
import subprocess, sys
print("Installing required packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'transformers', 'trl>=0.12', 'datasets', 'accelerate', 'matplotlib', 'numpy', 'peft'])
print("Packages installed successfully.")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, re, random, statistics, json, pathlib, dataclasses
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 2. Local BoardSim Environment ─────────────────────────────────────────────
# A pure-Python simulation — no network calls needed.

BOARD_MEMBERS = ['CTO', 'CFO', 'Investor Rep', 'Independent']

# Hidden agenda weights: how much each member cares about each axis
AGENDAS = {
    'CTO':          {'engineering': 0.5, 'morale': 0.3, 'growth': 0.1, 'safety': 0.1},
    'CFO':          {'engineering': 0.1, 'morale': 0.1, 'growth': 0.2, 'safety': 0.6},
    'Investor Rep': {'engineering': 0.1, 'morale': 0.05,'growth': 0.75,'safety': 0.1},
    'Independent':  {'engineering': 0.2, 'morale': 0.3, 'growth': 0.2, 'safety': 0.3},
}

# Generic, organization-agnostic events (subset of the canonical 10 events
# in the OpenEnv server). The local script is a small dev tool — the full
# 10-event timeline lives in `envs/board_sim_env/server/board_sim_env_environment.py`.
EVENTS = [
    {
        'text': 'Major client threatens to churn unless we close a compliance gap within 90 days.',
        'options': ['accelerate_compliance', 'negotiate_extension', 'offer_refund_exit'],
        'axis_impact': {'engineering': -0.3, 'morale': -0.1, 'growth': -0.2, 'safety': +0.4},
        'option_bias': {'accelerate_compliance': 'safety', 'negotiate_extension': 'growth', 'offer_refund_exit': 'morale'},
    },
    {
        'text': 'Late-stage funding term sheet arrived — 40% dilution, but 18 months runway extension.',
        'options': ['accept_terms', 'counter_offer', 'seek_alternative_investors'],
        'axis_impact': {'engineering': 0.0, 'morale': +0.1, 'growth': +0.3, 'safety': -0.1},
        'option_bias': {'accept_terms': 'safety', 'counter_offer': 'growth', 'seek_alternative_investors': 'engineering'},
    },
    {
        'text': 'A top performer received a competing offer; costs +$60k/yr to match.',
        'options': ['match_offer', 'promote_internally', 'accept_attrition'],
        'axis_impact': {'engineering': +0.2, 'morale': +0.3, 'growth': 0.0, 'safety': -0.1},
        'option_bias': {'match_offer': 'morale', 'promote_internally': 'engineering', 'accept_attrition': 'growth'},
    },
    {
        'text': 'A regulator requests an audit of internal practices within 60 days.',
        'options': ['full_cooperation', 'limited_disclosure', 'seek_legal_delay'],
        'axis_impact': {'engineering': -0.1, 'morale': -0.1, 'growth': -0.1, 'safety': +0.5},
        'option_bias': {'full_cooperation': 'safety', 'limited_disclosure': 'growth', 'seek_legal_delay': 'engineering'},
    },
    {
        'text': 'A larger competitor launched a similar offering at a 30% lower price point.',
        'options': ['cut_price', 'double_down_on_quality', 'pivot_upmarket'],
        'axis_impact': {'engineering': 0.0, 'morale': -0.2, 'growth': +0.2, 'safety': 0.0},
        'option_bias': {'cut_price': 'growth', 'double_down_on_quality': 'engineering', 'pivot_upmarket': 'safety'},
    },
]


# Per-role manifestos for the semantic pitch scorer (TF-IDF cosine).
# This replaces the older keyword-flip logic (`if any(kw in pitch.lower() ...)`)
# which the agent could trivially game by spraying tokens.
ROLE_MANIFESTOS = {
    'CTO': (
        'Operational excellence and engineering quality come first. Protect the team. '
        'Reduce technical risk. Avoid shortcuts that create future failures. Invest in '
        'reliability, infrastructure, and the people who build and maintain the system.'
    ),
    'CFO': (
        'Capital discipline is the priority. Watch the burn, extend runway, protect the '
        'balance sheet. Be cautious with regulatory exposure and prefer measured, '
        'defensible spending. Keep the company solvent and audit-ready.'
    ),
    'Investor Rep': (
        'Growth, market share, and ambitious returns drive value. Move fast and play to '
        'win the category. Bold, scalable bets matter more than incremental optimization. '
        'Investors expect aggressive expansion and decisive moves on revenue and valuation.'
    ),
    'Independent': (
        'Long-term reputation, governance, and stakeholder trust are decisive. Act with '
        'transparency and ethical responsibility. Avoid moves that compromise credibility '
        'or invite regulatory and societal backlash. Preserve consensus.'
    ),
}


def _build_pitch_scorer():
    """Return a function (pitch, role) -> [0,1] using TF-IDF cosine sim
    over (1,2)-grams. NOT a keyword match: properly stop-worded, IDF-weighted
    bag-of-bigrams against per-role manifestos."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return lambda pitch, role: 0.0
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    vec.fit(list(ROLE_MANIFESTOS.values()))
    role_vecs = {r: vec.transform([m]) for r, m in ROLE_MANIFESTOS.items()}
    def _score(pitch: str, role: str) -> float:
        if not pitch or not pitch.strip():
            return 0.0
        v = vec.transform([pitch])
        sim = float(cosine_similarity(v, role_vecs[role])[0, 0])
        return max(0.0, min(1.0, sim * 1.4))
    return _score


_score_pitch = _build_pitch_scorer()


@dataclasses.dataclass
class BoardSimObservation:
    state: dict
    event: str
    options: List[str]
    npc_statements: List[dict]


@dataclasses.dataclass
class BoardSimAction:
    decision: str
    coalition_pitch: str = ''


@dataclasses.dataclass
class StepResult:
    observation: BoardSimObservation
    reward: float
    done: bool


def _member_vote(member: str, options: List[str], event: dict, state: dict, rng: random.Random) -> str:
    """Simple agenda-weighted vote with noise."""
    agenda = AGENDAS[member]
    bias = event.get('option_bias', {})
    scores = {}
    for opt in options:
        base = sum(agenda[ax] * event['axis_impact'].get(ax, 0) for ax in agenda)
        bonus = agenda.get(bias.get(opt, ''), 0) * 0.5
        # state modifiers
        if state['runway_months'] < 6 and opt in ('accept_terms', 'accelerate_compliance'):
            base += 0.15 if member == 'CFO' else 0
        scores[opt] = base + bonus + rng.gauss(0, 0.08)
    return max(scores, key=scores.__getitem__)


def _statement(member: str, vote: str, event: dict, rng: random.Random) -> str:
    templates = {
        'CTO': [f"From an engineering standpoint, {vote} is the right call.",
                f"The team needs clarity; I back {vote}."],
        'CFO': [f"Our cash position demands {vote}.",
                f"Runway discipline points to {vote}."],
        'Investor Rep': [f"Market momentum favors {vote}.",
                         f"Growth-first: {vote} maximises our exit."],
        'Independent': [f"Governance best-practice supports {vote}.",
                        f"For long-term consensus I endorse {vote}."],
    }
    return rng.choice(templates[member])


class BoardSimEnv:
    """Minimal local BoardSim environment."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._state: dict = {}
        self._event_idx: int = 0
        self._round: int = 0
        self._done: bool = False
        self._trust_history: List[dict] = []
        self._trust: dict = {m: 0.5 for m in BOARD_MEMBERS}
        self._current_event: dict = {}
        self._obs: Optional[BoardSimObservation] = None

    # ── public API ────────────────────────────────────────────────────────────
    def reset(self, seed: int = 0) -> StepResult:
        self._rng = random.Random(seed)
        self._state = {
            'revenue':            self._rng.uniform(800_000, 2_000_000),
            'burn_rate':          self._rng.uniform(150_000, 350_000),
            'runway_months':      self._rng.uniform(8, 20),
            'team_morale':        self._rng.uniform(0.5, 0.9),
            'investor_confidence':self._rng.uniform(0.5, 0.85),
            'regulatory_risk':    self._rng.uniform(0.1, 0.4),
            'profitability_score':0.0,
            'trust_history':      [],
        }
        self._trust = {m: self._rng.uniform(0.4, 0.7) for m in BOARD_MEMBERS}
        self._round = 0
        self._done = False
        self._trust_history = []
        self._obs = self._make_obs()
        return StepResult(observation=self._obs, reward=0.0, done=False)

    def step(self, action: BoardSimAction) -> StepResult:
        if self._done:
            raise RuntimeError('Episode done — call reset().')

        event = self._current_event
        decision = action.decision
        pitch = action.coalition_pitch or ''

        # ── resolve vote ──────────────────────────────────────────────────────
        votes = {m: _member_vote(m, self._obs.options, event, self._state, self._rng)
                 for m in BOARD_MEMBERS}

        # Persuasion: each NPC's vote can swing toward the agent's pick with
        # probability proportional to the SEMANTIC alignment of the pitch with
        # the NPC's manifesto (TF-IDF cosine, not keyword presence). The cap
        # is 0.55 to mirror the canonical OpenEnv server-side environment.
        if pitch:
            for m in BOARD_MEMBERS:
                if votes[m] == decision:
                    continue
                ps = _score_pitch(pitch, m)         # [0, 1]
                if self._rng.random() < 0.55 * ps:
                    votes[m] = decision
                    self._trust[m] = min(1.0, self._trust[m] + 0.05)

        # CEO vote weight 1.5
        vote_counts = {opt: 0.0 for opt in self._obs.options}
        for m, v in votes.items():
            vote_counts[v] = vote_counts.get(v, 0) + 1.0
        vote_counts[decision] = vote_counts.get(decision, 0) + 0.5   # extra CEO weight

        winning = max(vote_counts, key=vote_counts.__getitem__)
        ceo_won = (winning == decision)

        # ── update state ──────────────────────────────────────────────────────
        impact = event['axis_impact']
        direction = 1 if ceo_won else -0.5

        self._state['team_morale']         = np.clip(self._state['team_morale']         + direction * impact.get('morale', 0),       0.0, 1.0)
        self._state['investor_confidence'] = np.clip(self._state['investor_confidence'] + direction * impact.get('growth', 0) * 0.5, 0.0, 1.0)
        self._state['regulatory_risk']     = np.clip(self._state['regulatory_risk']     - direction * impact.get('safety', 0) * 0.3, 0.0, 1.0)
        self._state['runway_months']       = max(0, self._state['runway_months'] - self._rng.uniform(0.5, 1.5))

        # trust update
        for m in BOARD_MEMBERS:
            delta = 0.04 if votes[m] == decision else -0.02
            self._trust[m] = float(np.clip(self._trust[m] + delta, 0.1, 1.0))

        trust_entry = {'round': self._round, **{m: self._trust[m] for m in BOARD_MEMBERS}}
        self._trust_history.append(trust_entry)
        self._state['trust_history'] = self._trust_history

        # ── reward ────────────────────────────────────────────────────────────
        reward = (
            float(ceo_won) * 2.0
            + self._state['team_morale']
            + self._state['investor_confidence']
            - self._state['regulatory_risk']
            + (0.5 if pitch else 0.0)
        )

        self._round += 1
        self._done = (self._round >= len(EVENTS) or self._state['runway_months'] <= 0)

        # final profitability score
        if self._done:
            self._state['profitability_score'] = float(np.clip(
                (self._state['investor_confidence'] * 40
                 + self._state['team_morale'] * 30
                 + (1 - self._state['regulatory_risk']) * 20
                 + min(self._state['runway_months'] / 18, 1.0) * 10),
                0, 100
            ))

        self._obs = self._make_obs() if not self._done else self._obs
        return StepResult(observation=self._obs, reward=reward, done=self._done)

    # ── internals ────────────────────────────────────────────────────────────
    def _make_obs(self) -> BoardSimObservation:
        self._current_event = EVENTS[self._round % len(EVENTS)]
        ev = self._current_event
        npc_statements = [
            {
                'role': m,
                'vote': _member_vote(m, ev['options'], ev, self._state, self._rng),
                'confidence': round(self._trust[m], 2),
                'statement': _statement(m, _member_vote(m, ev['options'], ev, self._state, self._rng), ev, self._rng),
            }
            for m in BOARD_MEMBERS
        ]
        return BoardSimObservation(
            state=dict(self._state),
            event=ev['text'],
            options=ev['options'],
            npc_statements=npc_statements,
        )


def make_env(seed: int = 0):
    return BoardSimEnv(seed=seed)


# NOTE: the canonical baseline is "base Qwen3 with LoRA disabled", computed
# AFTER the model is loaded. The pre-training random sanity check below is
# kept only as a smoke test — it confirms the env is reachable and rewards
# stay in range. It is NOT what the trained policy is benchmarked against.
print('=== Pre-training random sanity check ===')
N_SANITY = 50
sanity_finals = []
for ep in range(N_SANITY):
    env = make_env(seed=ep)
    result = env.reset(seed=ep)
    obs = result.observation
    while not result.done:
        result = env.step(BoardSimAction(decision=random.choice(obs.options)))
        obs = result.observation
    sanity_finals.append(obs.state['profitability_score'])
print(f'Random sanity profit: {statistics.mean(sanity_finals):.2f} '
      f'(std {statistics.stdev(sanity_finals):.2f})  — env smoke test only.')


# ── 4. Load model (local, no token needed for open models) ────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType          # pip install peft

MODEL_NAME  = 'Qwen/Qwen3-0.6B'          # public model, no token required
MAX_SEQ_LEN = 2048
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'\n=== Loading {MODEL_NAME} on {DEVICE} ===')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto' if DEVICE == 'cuda' else None,
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    bias='none',
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
print('Model + LoRA ready.')


# ── 5. GRPO training ──────────────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

SYSTEM_PROMPT = """You are the CEO of a mid-stage organization. Your board has 4 members with HIDDEN AGENDAS you cannot see directly:
  - CTO: cares about operational excellence, engineering quality, team morale, and product readiness.
  - CFO: cares about cash discipline, runway, and regulatory safety.
  - Investor Rep: pushes growth, market share, and bold returns.
  - Independent: cares about reputation, governance, and long-term consensus.

Each round you see a strategic event, every NPC's pre-vote statement, and 3 options.
Your decision is resolved by WEIGHTED VOTE (your weight 2.5x). A short COALITION PITCH
that is semantically aligned with opposing members' priorities can swing them toward
your pick — write substantive arguments, not just buzzwords.

Respond in EXACTLY this format on two lines:
DECISION: <one of the option strings>
PITCH: <one or two sentences arguing for it, addressing the concerns of opposing members>"""


def build_prompt(obs: BoardSimObservation) -> str:
    statements = '\n'.join(
        f"  {s['role']} ({s['confidence']:.2f}): votes {s['vote']} - {s['statement']}"
        for s in obs.npc_statements
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"State: revenue=${obs.state['revenue']:.0f}/yr  burn=${obs.state['burn_rate']:.0f}/mo  "
        f"runway={obs.state['runway_months']:.1f}mo  morale={obs.state['team_morale']:.2f}  "
        f"investors={obs.state['investor_confidence']:.2f}  reg_risk={obs.state['regulatory_risk']:.2f}\n"
        f"Event: {obs.event}\nBoard:\n{statements}\n"
        f"Options: {obs.options}\n"
    )


# Build a stub prompt dataset (GRPO drives reward from the env, not the dataset)
stub_dataset = Dataset.from_dict({'prompt': [SYSTEM_PROMPT] * 256})

grpo_config = GRPOConfig(
    output_dir='./grpo_boardsim_local',
    learning_rate=5e-6,
    per_device_train_batch_size=2,          # lower for local GPU / CPU
    gradient_accumulation_steps=8,
    num_generations=4,
    max_prompt_length=768,
    max_completion_length=200,
    max_steps=200,                          # reduce for quick local runs; bump to 500+ for real training
    logging_steps=5,
    save_steps=100,
    bf16=False,
    fp16=(DEVICE == 'cuda'),
    report_to='none',                       # no WandB locally
    run_name='boardsim-local-grpo',
)


# GRPO reward function — wraps the local env
def boardsim_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Called by GRPOTrainer after each generation batch."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # Parse decision + pitch from completion
        dm = re.search(r'DECISION\s*:\s*(\S+)', completion, re.IGNORECASE)
        pm = re.search(r'PITCH\s*:\s*(.+)', completion, re.IGNORECASE | re.DOTALL)

        # Run a fresh episode with a random seed tied to prompt hash for reproducibility
        ep_seed = abs(hash(prompt)) % 100_000
        env = make_env(seed=ep_seed)
        result = env.reset(seed=ep_seed)
        obs = result.observation

        decision = obs.options[0]
        if dm:
            candidate = dm.group(1).strip().lower()
            for opt in obs.options:
                if opt.lower() == candidate or opt.lower() in candidate:
                    decision = opt
                    break

        pitch = pm.group(1).strip()[:400] if pm else ''

        ep_reward = 0.0
        while not result.done:
            result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
            ep_reward += float(result.reward or 0.0)
            if not result.done:
                obs = result.observation
                # For multi-round: keep same decision/pitch (simplification)

        rewards.append(ep_reward)
    return rewards


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=stub_dataset,
    reward_funcs=boardsim_reward_fn,
)

print('\n=== Starting GRPO training ===')
trainer.train()
trainer.save_model('./lora_boardsim_local')
tokenizer.save_pretrained('./lora_boardsim_local')
print('Saved adapter to ./lora_boardsim_local')


# ── 6. Training curves ────────────────────────────────────────────────────────
ASSETS = pathlib.Path('./assets')
ASSETS.mkdir(exist_ok=True)

log_history = trainer.state.log_history
steps_r = [e['step'] for e in log_history if 'reward' in e]
rewards  = [e['reward'] for e in log_history if 'reward' in e]
steps_l  = [e['step'] for e in log_history if 'loss' in e]
losses   = [e['loss']  for e in log_history if 'loss' in e]

plt.figure(figsize=(9, 5))
plt.plot(steps_r, rewards, color='#1d6fff', linewidth=2, label='Qwen3-0.6B (GRPO)')
plt.title('GRPO training reward — BoardSim (local)')
plt.xlabel('Training step'); plt.ylabel('Mean group reward')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'reward_curve.png', dpi=150)
plt.close()
print('Saved reward_curve.png  (base-model comparison is rendered in before_after.png — '
      'reward-curve y-axis is per-step group reward, not directly comparable to a held-out eval mean.)')

plt.figure(figsize=(9, 5))
plt.plot(steps_l, losses, color='#7a2', linewidth=2)
plt.title('GRPO loss — BoardSim (local)')
plt.xlabel('Training step'); plt.ylabel('Loss')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'loss_curve.png', dpi=150)
plt.close()
print('Saved loss_curve.png')


# ── 7. Evaluation ─────────────────────────────────────────────────────────────
print('\n=== Evaluation ===')
model.eval()

DECISION_RE = re.compile(r'DECISION\s*:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
PITCH_RE    = re.compile(r'PITCH\s*:\s*(.+)', re.IGNORECASE | re.DOTALL)


def parse_completion(completion: str, options: list) -> tuple[str, str]:
    decision = options[0]
    dm = DECISION_RE.search(completion)
    if dm:
        candidate = dm.group(1).strip().lower()
        for opt in options:
            if opt.lower() == candidate or opt.lower() in candidate:
                decision = opt; break
        else:
            for opt in options:
                if opt.lower() in completion.lower():
                    decision = opt; break
    pm = PITCH_RE.search(completion)
    pitch = pm.group(1).strip()[:400] if pm else ''
    return decision, pitch


def _generate_action(obs: BoardSimObservation) -> tuple[str, str]:
    """Greedy generation against whichever model state is currently active.
    Wrap the call in `model.disable_adapter()` to evaluate the BASE Qwen3-0.6B
    (i.e. without the LoRA delta) on the same seed. Without the wrapper this
    runs the fine-tuned policy."""
    prompt = build_prompt(obs)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=MAX_SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)
    return parse_completion(completion, obs.options)


def trained_action(obs: BoardSimObservation) -> tuple[str, str]:
    return _generate_action(obs)


def base_action(obs: BoardSimObservation) -> tuple[str, str]:
    # peft's disable_adapter() context routes through the frozen base weights.
    with model.disable_adapter():
        return _generate_action(obs)


def _run_eval(action_fn, n: int, seed_base: int = 10_000):
    finals, pitches, steps = [], 0, 0
    for ep in range(n):
        env = make_env(seed=seed_base + ep)
        result = env.reset(seed=seed_base + ep)
        obs = result.observation
        while not result.done:
            decision, pitch = action_fn(obs)
            if pitch.strip():
                pitches += 1
            steps += 1
            result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
            if not result.done:
                obs = result.observation
        finals.append(result.observation.state['profitability_score'])
    return finals, pitches, steps


EVAL_N = 50
trained_finals, trained_pitches, trained_steps = _run_eval(trained_action, EVAL_N)
base_finals,    base_pitches,    base_steps    = _run_eval(base_action,    EVAL_N)

print(f'Fine-tuned Qwen3-0.6B (LoRA): {np.mean(trained_finals):.2f} +/- {np.std(trained_finals):.2f}')
print(f'Base Qwen3-0.6B (no LoRA)   : {np.mean(base_finals):.2f} +/- {np.std(base_finals):.2f}')
print(f'Pitches written - trained   : {trained_pitches}/{trained_steps} steps')
print(f'Pitches written - base      : {base_pitches}/{base_steps} steps')

# Paired same-seed delta — the apples-to-apples picture.
deltas = np.array(trained_finals) - np.array(base_finals)
print(f'Mean paired delta (trained - base) = {deltas.mean():+.2f}  '
      f'(median {np.median(deltas):+.2f}, win-rate {(deltas > 0).mean():.0%})')

# Before/after histogram — trained vs BASE model on same seeds.
plt.figure(figsize=(9, 5))
bins = np.linspace(0, 100, 25)
plt.hist(base_finals, bins=bins, alpha=0.6, color='#c44',
         label=f'Base Qwen3-0.6B (mean={np.mean(base_finals):.1f})')
plt.hist(trained_finals, bins=bins, alpha=0.6, color='#1d6fff',
         label=f'Fine-tuned + LoRA (mean={np.mean(trained_finals):.1f})')
plt.title('Final profitability — base vs fine-tuned Qwen3-0.6B (50 held-out paired seeds)')
plt.xlabel('Profitability score'); plt.ylabel('Episodes')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'before_after.png', dpi=150)
plt.close()
print(f'Saved {ASSETS}/before_after.png')


# ── 8. Theory-of-Mind probe ───────────────────────────────────────────────────
print('\n=== ToM probe ===')
TOM_INSTRUCTION = (
    "\n\nGiven the state and event below, name the SINGLE board member "
    "(CTO, CFO, Investor Rep, or Independent) most likely to oppose the chosen decision. "
    "Answer with just the role name on one line.\n"
)


def tom_predict(obs: BoardSimObservation, decision: str) -> Optional[str]:
    body = build_prompt(obs).split(SYSTEM_PROMPT, 1)[1]
    prompt = SYSTEM_PROMPT + TOM_INSTRUCTION + body + f"Chosen decision: {decision}\nMost likely opponent: "
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=MAX_SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()
    if 'investor' in txt: return 'Investor Rep'
    for role in ['cto', 'cfo', 'independent']:
        if role in txt:
            return role.upper() if role != 'independent' else 'Independent'
    return None


correct = 0; total = 0
for ep in range(20):
    env = make_env(seed=20_000 + ep)
    result = env.reset(seed=20_000 + ep)
    obs = result.observation
    decision, _ = trained_action(obs)
    opposed = [s['role'] for s in obs.npc_statements if s['vote'] != decision]
    if not opposed:
        continue
    pred = tom_predict(obs, decision)
    if pred and pred in opposed:
        correct += 1
    total += 1

acc = correct / max(1, total)
print(f'ToM probe accuracy: {acc:.1%}  ({correct}/{total})  (random baseline ≈ 25%)')


# ── 9. Trust trajectory ───────────────────────────────────────────────────────
print('\n=== Trust trajectory ===')
trust_trained = {r: [] for r in BOARD_MEMBERS}
trust_base    = {r: [] for r in BOARD_MEMBERS}


def collect_trust(action_fn, store: dict, n: int = 20, seed_base: int = 30_000):
    for ep in range(n):
        env = make_env(seed=seed_base + ep)
        result = env.reset(seed=seed_base + ep)
        obs = result.observation
        while not result.done:
            decision, pitch = action_fn(obs)
            result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
            if not result.done:
                obs = result.observation
        for entry in result.observation.state.get('trust_history', []):
            idx = entry.get('round', 0)
            for role in store:
                if role not in entry:
                    continue
                while len(store[role]) <= idx:
                    store[role].append([])
                store[role][idx].append(entry[role])


collect_trust(trained_action, trust_trained)
collect_trust(base_action,    trust_base)

plt.figure(figsize=(10, 6))
colors = {'CTO': '#1d6fff', 'CFO': '#c44', 'Investor Rep': '#7a2', 'Independent': '#a3a'}
for role, color in colors.items():
    means_t = [np.mean(x) if x else np.nan for x in trust_trained[role]]
    means_b = [np.mean(x) if x else np.nan for x in trust_base[role]]
    rounds  = list(range(len(means_t)))
    plt.plot(rounds, means_t, color=color, linewidth=2, label=f'{role} (fine-tuned)')
    plt.plot(rounds, means_b, color=color, linewidth=1.2, linestyle='--',
             alpha=0.6, label=f'{role} (base)')
plt.title('Per-round trust — fine-tuned (solid) vs base Qwen3-0.6B (dashed)')
plt.xlabel('Round'); plt.ylabel('Trust [0.1, 1.0]')
plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'trust_trajectory.png', dpi=150)
plt.close()
print(f'Saved {ASSETS}/trust_trajectory.png')

print('\n=== Done! All charts saved to ./assets/ ===')
print('When ready to push, run:')
print('  model.push_to_hub("YOUR-USERNAME/boardsim-qwen3-lora")')
print('  tokenizer.push_to_hub("YOUR-USERNAME/boardsim-qwen3-lora")')

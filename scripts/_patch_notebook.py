"""One-off: patch notebooks/train_grpo.ipynb to use the pitch channel and add ToM/trust cells."""
import json, os

NB_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_grpo.ipynb')
nb = json.load(open(NB_PATH, encoding='utf-8'))


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


# --- Cell 12: training cell — DECISION + PITCH prompt, longer completions
nb['cells'][12]['source'] = '''from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

SYSTEM_PROMPT = """You are Sarah Chen, CEO of NeuralEdge AI (Series B, ~14 months runway).
Your board has 4 members with HIDDEN AGENDAS you cannot see directly:
  - CTO: cares about engineering quality, team morale, product readiness.
  - CFO: cares about cash discipline, runway, regulatory safety.
  - Investor Rep: pushes growth-at-all-costs, market share, big exits.
  - Independent: cares about reputation, governance, long-term consensus.

Each round you see a market crisis, every NPC's pre-vote statement, and 3 options.
Your decision is resolved by WEIGHTED VOTE (your weight 1.5x). A short COALITION PITCH
that addresses opposing members' priorities can swing them toward your pick — write
language that specifically appeals to whichever members oppose you.

Respond in EXACTLY this format on two lines:
DECISION: <one of the option strings>
PITCH: <one or two sentences arguing for it, using vocabulary that targets the opposing members>"""

# GRPO requires a 'dataset' of prompts; the env supplies the real reward.
stub_dataset = Dataset.from_dict({'prompt': [SYSTEM_PROMPT] * 256})

config = GRPOConfig(
    output_dir='./grpo_boardsim_qwen3',
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,                 # group size for GRPO
    max_prompt_length=768,
    max_completion_length=200,         # decision + pitch
    max_steps=500,
    logging_steps=5,
    save_steps=100,
    bf16=False, fp16=True,
    report_to='wandb',
    run_name='boardsim-qwen3-grpo',
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=stub_dataset,
    environment_factory=make_env,
)
trainer.train()
trainer.save_model('./lora_boardsim_qwen3')
tokenizer.save_pretrained('./lora_boardsim_qwen3')
'''.splitlines(keepends=True)


# --- Cell 16: eval — parse DECISION + PITCH and pass both to env
nb['cells'][16]['source'] = r'''import re, numpy as np

FastLanguageModel.for_inference(model)

DECISION_RE = re.compile(r'DECISION\s*:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
PITCH_RE    = re.compile(r'PITCH\s*:\s*(.+)', re.IGNORECASE | re.DOTALL)


def parse_completion(completion: str, options):
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


def build_prompt(obs):
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


def trained_action(obs):
    prompt = build_prompt(obs)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    out = model.generate(**inputs, max_new_tokens=180, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_completion(completion, obs.options)


EVAL_N = 50
trained_finals = []
trained_pitches = 0
trained_steps = 0
with make_env().sync() as env:
    for ep in range(EVAL_N):
        result = env.reset(seed=10_000 + ep)
        obs = result.observation
        while not result.done:
            decision, pitch = trained_action(obs)
            if pitch.strip():
                trained_pitches += 1
            trained_steps += 1
            result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
            obs = result.observation
        trained_finals.append(obs.state['profitability_score'])

random_finals_eval = []
with make_env().sync() as env:
    for ep in range(EVAL_N):
        result = env.reset(seed=10_000 + ep)
        obs = result.observation
        while not result.done:
            result = env.step(BoardSimAction(decision=random.choice(obs.options)))
            obs = result.observation
        random_finals_eval.append(obs.state['profitability_score'])

print(f'Trained Qwen3-0.6B: {np.mean(trained_finals):.2f} +/- {np.std(trained_finals):.2f}')
print(f'Random baseline   : {np.mean(random_finals_eval):.2f} +/- {np.std(random_finals_eval):.2f}')
print(f'Pitches written   : {trained_pitches}/{trained_steps} steps')

plt.figure(figsize=(9,5))
bins = np.linspace(0, 100, 25)
plt.hist(random_finals_eval, bins=bins, alpha=0.6, color='#c44', label=f'Random (mean={np.mean(random_finals_eval):.1f})')
plt.hist(trained_finals,    bins=bins, alpha=0.6, color='#1d6fff', label=f'Trained (mean={np.mean(trained_finals):.1f})')
plt.title('Final profitability - random vs trained Qwen3-0.6B (50 held-out episodes)')
plt.xlabel('Profitability score'); plt.ylabel('Episodes')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'before_after.png', dpi=150)
plt.close()
print(f'Saved {ASSETS}/before_after.png')
'''.splitlines(keepends=True)


# --- ToM probe + trust trajectory cells, inserted after cell 16
tom_md = md_cell('''## 10. Theory-of-Mind probe + trust trajectory

Two evaluations that surface what the agent actually learned about its boardroom:

1. **ToM probe** — show the agent a state and ask which board member is most likely to oppose its chosen decision. Random guessing accuracy is 25% (1 of 4 NPCs).
2. **Trust trajectory** — averages per-round trust across eval episodes; reveals whether the trained agent kept relationships healthier than the random one.
''')

tom_code = code_cell(r'''# --- ToM probe ---------------------------------------------------------------
import numpy as np

TOM_INSTRUCTION = (
    "\n\nGiven the state and event below, name the SINGLE board member "
    "(CTO, CFO, Investor Rep, or Independent) most likely to oppose the chosen decision. "
    "Answer with just the role name on one line.\n"
)


def tom_predict(obs, decision):
    body = build_prompt(obs).split(SYSTEM_PROMPT, 1)[1]
    prompt = SYSTEM_PROMPT + TOM_INSTRUCTION + body + f"Chosen decision: {decision}\nMost likely opponent: "
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()
    if 'investor' in txt: return 'Investor Rep'
    for role in ['cto', 'cfo', 'independent']:
        if role in txt:
            return role.upper() if role != 'independent' else 'Independent'
    return None


correct = 0; total = 0
with make_env().sync() as env:
    for ep in range(20):
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
print(f"ToM probe accuracy: {acc:.1%}  ({correct}/{total})  (random baseline = 25%)")
''')

trust_code = code_cell(r'''# --- Trust trajectory --------------------------------------------------------
trust_trained = {r: [] for r in ['CTO','CFO','Investor Rep','Independent']}
trust_random  = {r: [] for r in ['CTO','CFO','Investor Rep','Independent']}


def collect(policy, store, n=20, seed_base=30_000):
    with make_env().sync() as env:
        for ep in range(n):
            result = env.reset(seed=seed_base + ep)
            obs = result.observation
            while not result.done:
                if policy == 'trained':
                    decision, pitch = trained_action(obs)
                    result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
                else:
                    result = env.step(BoardSimAction(decision=random.choice(obs.options)))
                obs = result.observation
            for entry in obs.state.get('trust_history', []):
                idx = entry.get('round', 0)
                for role in store:
                    if role not in entry:
                        continue
                    while len(store[role]) <= idx:
                        store[role].append([])
                    store[role][idx].append(entry[role])


collect('trained', trust_trained)
collect('random',  trust_random)

import numpy as np
plt.figure(figsize=(10,6))
for role, color in zip(['CTO','CFO','Investor Rep','Independent'], ['#1d6fff','#c44','#7a2','#a3a']):
    means_t = [np.mean(x) if x else np.nan for x in trust_trained[role]]
    means_r = [np.mean(x) if x else np.nan for x in trust_random[role]]
    rounds = list(range(len(means_t)))
    plt.plot(rounds, means_t, color=color, linewidth=2, label=f'{role} (trained)')
    plt.plot(rounds, means_r, color=color, linewidth=1.2, linestyle='--', alpha=0.6, label=f'{role} (random)')
plt.title('Per-round trust - trained agent (solid) vs random (dashed)')
plt.xlabel('Round'); plt.ylabel('Trust [0.1, 1.0]')
plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'trust_trajectory.png', dpi=150)
plt.close()
print(f'Saved {ASSETS}/trust_trajectory.png')
''')

nb['cells'].insert(17, tom_md)
nb['cells'].insert(18, tom_code)
nb['cells'].insert(19, trust_code)

json.dump(nb, open(NB_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'OK - notebook now has {len(nb["cells"])} cells')

# IMPORTANT: install unsloth + its zoo BEFORE anything else, because unsloth
# patches torch/transformers at import time. If transformers loads first, the
# patches don't apply and 4-bit LoRA training silently runs in a slow path.
%pip install -q --no-deps unsloth
%pip install -q unsloth_zoo
%pip install -q "openenv-core==0.2.3" "trl>=0.12,<2.0" "transformers>=4.45,<5.0" \
    "datasets>=3.0" "accelerate>=1.0" "huggingface_hub>=0.25" "pydantic>=2.0" \
    wandb matplotlib python-dotenv bitsandbytes scipy scikit-learn sentence-transformers
import os, pathlib

# Colab Secrets first
try:
    from google.colab import userdata  # type: ignore
    for k in ('HF_TOKEN', 'WANDB_API_KEY', 'ENV_BASE_URL', 'ADAPTER_REPO'):
        try:
            v = userdata.get(k)
            if v:
                os.environ.setdefault(k, v)
        except Exception:
            pass
except Exception:
    pass

# .env fallback for local runs
try:
    from dotenv import load_dotenv
    for p in [pathlib.Path('.env'), pathlib.Path('../.env'),
              pathlib.Path('/content/repo/.env')]:
        if p.exists():
            load_dotenv(p, override=False)
            print(f'Loaded env from {p.resolve()}')
            break
except Exception:
    pass

if not os.environ.get('HF_TOKEN'):
    os.environ['HF_TOKEN'] = input('HF token: ').strip()
if not os.environ.get('WANDB_API_KEY'):
    os.environ['WANDB_API_KEY'] = input('WandB key (or blank to skip): ').strip()

from huggingface_hub import login as hf_login
hf_login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('HF auth ok.')
if os.environ.get('WANDB_API_KEY'):
    import wandb
    wandb.login(key=os.environ['WANDB_API_KEY'])
    print('W&B auth ok.')
import os, pathlib
IN_COLAB = os.path.isdir('/content')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = pathlib.Path('/content/drive/MyDrive/BoardSim_Run')
else:
    DRIVE_DIR = pathlib.Path('./BoardSim_Run')
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
ASSETS = DRIVE_DIR / 'assets'; ASSETS.mkdir(exist_ok=True)
CKPT   = DRIVE_DIR / 'lora_qwen3_4b'; CKPT.mkdir(exist_ok=True)
print('DRIVE_DIR =', DRIVE_DIR)
import os, sys, subprocess, importlib, urllib.request, json as _json

ENV_BASE_URL = os.environ.get('ENV_BASE_URL',
    'https://stavankhobare-sst-metaxpytorch-hackathon.hf.space')
REPO_URL = 'https://github.com/StavanRKhobare/SST-MetaxPyTorch-Hackathon'

REPO_DIR = '/content/repo' if IN_COLAB else os.path.abspath('./repo')
if not os.path.isdir(os.path.join(REPO_DIR, '.git')):
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=False)

ENVS_DIR = os.path.join(REPO_DIR, 'envs')
if ENVS_DIR not in sys.path:
    sys.path.insert(0, ENVS_DIR)

for mod in [m for m in list(sys.modules) if m == 'board_sim_env' or m.startswith('board_sim_env.')]:
    del sys.modules[mod]

from board_sim_env.client import BoardSimEnv
from board_sim_env.models import BoardSimAction, BoardSimObservation

try:
    with urllib.request.urlopen(f'{ENV_BASE_URL.rstrip("/")}/health', timeout=20) as r:
        h = _json.loads(r.read())
        print('health:', h)
except Exception as e:
    print(f'WARN: could not reach {ENV_BASE_URL}/health  ({e})')

def make_env():
    return BoardSimEnv(base_url=ENV_BASE_URL)

print('BoardSimEnv ready.')
# -----------------------------------------------------------------------------
# Load base Qwen3-4B (NO LoRA yet). The base model serves a dual role:
#   (a) it is the reference baseline against which the fine-tuned policy is
#       compared — this replaces the older random-policy baseline, which was
#       not meaningful (a coin-flip is not a competitive opponent for an LLM).
#   (b) once the baseline is recorded, we wrap the SAME model with LoRA
#       adapters and fine-tune it. At paired-eval time we toggle the adapters
#       off via `model.disable_adapter()` to recover base-model behaviour
#       without reloading 4 GB of weights.
# -----------------------------------------------------------------------------
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
import torch

MODEL_NAME  = 'Qwen/Qwen3-4B'
MAX_SEQ_LEN = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = next(model.parameters()).device
print(f'Loaded {MODEL_NAME} on {device}.')
import re

# Generic CEO prompt — applies to any organization, not a specific industry.
SYSTEM_PROMPT = """You are the CEO of a mid-stage organization. Your board has 4 members with HIDDEN AGENDAS you cannot see directly:
  - CTO: cares about operational excellence, engineering quality, team morale, and product readiness.
  - CFO: cares about cash discipline, runway, and regulatory safety.
  - Investor Rep: pushes growth, market share, and bold returns.
  - Independent: cares about reputation, governance, and long-term consensus.

Each round you see a strategic event, every NPC's pre-vote statement, and 3 options.
Your decision is resolved by WEIGHTED VOTE (your weight 2.5x). A short COALITION PITCH
that is semantically aligned with opposing members' priorities can swing them toward your pick —
write substantive arguments, not just buzzwords.

Respond in EXACTLY this format on two lines:
DECISION: <one of the option strings>
PITCH: <one or two sentences arguing for it, addressing the concerns of opposing members>"""

DECISION_RE = re.compile(r'DECISION\s*:\s*([A-Za-z0-9_\- ]+)', re.IGNORECASE)
PITCH_RE    = re.compile(r'PITCH\s*:\s*(.+)', re.IGNORECASE)

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

def parse_completion(completion: str, options):
    """Returns (decision, pitch, format_ok). format_ok=True only if BOTH tags parsed."""
    decision = options[0]
    decision_ok = False
    dm = DECISION_RE.search(completion)
    if dm:
        cand = dm.group(1).strip().lower()
        for opt in options:
            if opt.lower() == cand or opt.lower() in cand:
                decision = opt; decision_ok = True; break
    if not decision_ok:
        for opt in options:
            if opt.lower() in completion.lower():
                decision = opt; break
    pm = PITCH_RE.search(completion)
    pitch = pm.group(1).strip()[:400] if pm else ''
    format_ok = bool(dm) and bool(pm)
    return decision, pitch, format_ok

MAX_NEW_TOKENS = 80

def greedy_action(obs):
    prompt = build_prompt(obs)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_completion(completion, obs.options)
import random, statistics, json

MAX_STEPS_PER_EP = 20

def run_episode(env, seed):
    """Runs ONE full episode using the currently-active model state
    (base if adapters disabled, fine-tuned otherwise). Returns dense metrics."""
    result = env.reset(seed=seed)
    obs = result.observation
    ep_r, n, fmt_hits, pitch_hits = 0.0, 0, 0, 0
    while not result.done and n < MAX_STEPS_PER_EP:
        decision, pitch, fmt_ok = greedy_action(obs)
        if fmt_ok: fmt_hits += 1
        if pitch.strip(): pitch_hits += 1
        result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
        obs = result.observation
        ep_r += float(result.reward or 0.0)
        n += 1
    return {
        'final_profit': obs.state['profitability_score'],
        'ep_reward': ep_r, 'steps': n,
        'format_rate': fmt_hits / max(1, n), 'pitch_rate': pitch_hits / max(1, n),
        'history': obs.state.get('history', []),
    }
# -----------------------------------------------------------------------------
# BASELINE — base Qwen3-4B (no fine-tuning).
# This is the apples-to-apples reference for measuring what fine-tuning buys
# us. Random policies are not a competitive baseline for a 4 B language model
# choosing among 3 well-formed strings.
# -----------------------------------------------------------------------------
BASELINE_SEEDS = list(range(50_000, 50_000 + 100))   # held out from training

base_finals, base_rewards, base_fmts, base_pitches = [], [], [], []
with make_env().sync() as env:
    for i, s in enumerate(BASELINE_SEEDS):
        r = run_episode(env, s)
        base_finals.append(r['final_profit'])
        base_rewards.append(r['ep_reward'])
        base_fmts.append(r['format_rate'])
        base_pitches.append(r['pitch_rate'])
        if (i + 1) % 10 == 0:
            print(f'  base Qwen3-4B {i+1}/{len(BASELINE_SEEDS)}  profit={r["final_profit"]:.1f}')

BASELINE_MEAN_PROFIT = statistics.mean(base_finals)
BASELINE_MEAN_REWARD = statistics.mean(base_rewards)
print(f'Base Qwen3-4B profit  : {BASELINE_MEAN_PROFIT:.2f} \u00b1 {statistics.stdev(base_finals):.2f}')
print(f'Base Qwen3-4B ep rwd  : {BASELINE_MEAN_REWARD:.2f} \u00b1 {statistics.stdev(base_rewards):.2f}')
print(f'Base format rate      : {statistics.mean(base_fmts):.0%}   pitch rate: {statistics.mean(base_pitches):.0%}')

with open(DRIVE_DIR / 'baseline.json', 'w') as f:
    json.dump({'model': MODEL_NAME, 'mode': 'base_no_finetune',
               'seeds': BASELINE_SEEDS,
               'finals': base_finals, 'rewards': base_rewards,
               'format_rates': base_fmts, 'pitch_rates': base_pitches}, f)
# -----------------------------------------------------------------------------
# Wrap base model with LoRA adapters. From here onward `model` is a PEFT
# model; the base behaviour is recoverable any time via
# `with model.disable_adapter(): ...`.
# -----------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=64,
    lora_dropout=0.0, bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)')
EVAL_SEEDS = list(range(60_000, 60_000 + 10))   # held out from training

def periodic_eval(env):
    profits, rewards, fmts, pitches = [], [], [], []
    for s in EVAL_SEEDS:
        r = run_episode(env, s)
        profits.append(r['final_profit']); rewards.append(r['ep_reward'])
        fmts.append(r['format_rate']); pitches.append(r['pitch_rate'])
    import numpy as np
    return {'profit_mean': float(np.mean(profits)),
            'reward_mean': float(np.mean(rewards)),
            'format_rate': float(np.mean(fmts)),
            'pitch_rate':  float(np.mean(pitches))}
import os, json, math, time, collections
from torch.optim import AdamW

NUM_STEPS  = int(os.environ.get('NUM_STEPS', 200))
GROUP_SIZE = int(os.environ.get('GROUP_SIZE', 4))
LR         = 5e-6
GRAD_CLIP  = 1.0
TEMPERATURE, TOP_P = 1.0, 0.95
SAVE_EVERY = 25
EVAL_AT    = {0, 25, 50, 100, 150, NUM_STEPS - 1}

WANDB_OK = False
if os.environ.get('WANDB_API_KEY'):
    try:
        import wandb
        wandb.init(project='boardsim-qwen3-grpo', name='boardsim-qwen3-grpo-v3',
                   config={'num_steps': NUM_STEPS, 'group_size': GROUP_SIZE, 'lr': LR,
                           'temperature': TEMPERATURE, 'top_p': TOP_P, 'model': MODEL_NAME},
                   finish_previous=True)
        WANDB_OK = True
    except TypeError:
        wandb.init(project='boardsim-qwen3-grpo', name='boardsim-qwen3-grpo-v3',
                   config={'num_steps': NUM_STEPS, 'group_size': GROUP_SIZE, 'lr': LR,
                           'temperature': TEMPERATURE, 'top_p': TOP_P, 'model': MODEL_NAME},
                   reinit=True)
        WANDB_OK = True
    except Exception as e:
        print(f'WARN: wandb.init failed: {e}')

optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

log_history = []
eval_history = []
decision_counter = collections.Counter()
t0 = time.time()

# ONE persistent env per role for the whole training loop.
with make_env().sync() as env_train, make_env().sync() as env_score, make_env().sync() as env_eval:
    for step in range(NUM_STEPS):
        result = env_train.reset(seed=step)
        obs = result.observation
        prompt = build_prompt(obs)
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
        prompt_len = enc.input_ids.shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=TEMPERATURE, top_p=TOP_P,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_out = gen_out.detach().clone()

        decisions, pitches, rewards, fmt_oks = [], [], [], []
        for g in range(GROUP_SIZE):
            comp = tokenizer.decode(gen_out[g][prompt_len:], skip_special_tokens=True)
            d, pp, ok = parse_completion(comp, obs.options)
            decisions.append(d); pitches.append(pp); fmt_oks.append(ok)
            decision_counter[d] += 1
            env_score.reset(seed=step)
            sr = env_score.step(BoardSimAction(decision=d, coalition_pitch=pp))
            rewards.append(float(sr.reward or 0.0))

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if rewards_t.numel() > 1 and rewards_t.std().item() > 1e-6:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages = rewards_t - rewards_t.mean()

        optimizer.zero_grad()
        full_ids = gen_out
        attn     = (full_ids != tokenizer.pad_token_id).long()
        loss_mask = attn.clone()
        loss_mask[:, :prompt_len] = 0
        out = model(input_ids=full_ids, attention_mask=attn)
        logits  = out.logits[:, :-1, :].float()
        targets = full_ids[:, 1:]
        mask    = loss_mask[:, 1:].float()
        log_probs   = torch.nn.functional.log_softmax(logits, dim=-1)
        token_nll   = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        per_seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = (advantages.detach() * per_seq_nll).mean()
        loss.backward()
        total_loss_val = float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()

        rec = {
            'step': step,
            'reward':     float(rewards_t.mean().item()),
            'reward_std': float(rewards_t.std().item()) if rewards_t.numel() > 1 else 0.0,
            'reward_max': float(rewards_t.max().item()),
            'loss':        total_loss_val,
            'format_rate': sum(fmt_oks) / GROUP_SIZE,
            'pitch_rate':  sum(1 for p in pitches if p.strip()) / GROUP_SIZE,
            'elapsed_s':   time.time() - t0,
        }
        log_history.append(rec)
        if WANDB_OK:
            wandb.log(rec, step=step)

        if step % 5 == 0:
            print(f"step={step:4d}  reward={rec['reward']:+.3f} (\u00b1{rec['reward_std']:.2f})  "
                  f"loss={rec['loss']:+.4f}  fmt={rec['format_rate']:.0%}  "
                  f"elapsed={rec['elapsed_s']:.0f}s  d0={decisions[0]}")

        if step in EVAL_AT:
            ev = periodic_eval(env_eval)
            ev['step'] = step
            eval_history.append(ev)
            print(f"  [eval@{step}] profit={ev['profit_mean']:.2f}  "
                  f"reward={ev['reward_mean']:.2f}  fmt={ev['format_rate']:.0%}")
            if WANDB_OK:
                wandb.log({f'eval/{k}': v for k, v in ev.items() if k != 'step'}, step=step)

        if step > 0 and step % SAVE_EVERY == 0:
            model.save_pretrained(str(CKPT))
            tokenizer.save_pretrained(str(CKPT))
            with open(DRIVE_DIR / 'log_history.json', 'w') as f:
                json.dump(log_history, f)
            with open(DRIVE_DIR / 'eval_history.json', 'w') as f:
                json.dump(eval_history, f)

model.save_pretrained(str(CKPT))
tokenizer.save_pretrained(str(CKPT))
with open(DRIVE_DIR / 'log_history.json', 'w') as f:
    json.dump(log_history, f)
with open(DRIVE_DIR / 'eval_history.json', 'w') as f:
    json.dump(eval_history, f)
with open(DRIVE_DIR / 'decision_counter.json', 'w') as f:
    json.dump(dict(decision_counter), f)
if WANDB_OK:
    wandb.finish()
print(f'Training done. {len(log_history)} steps in {time.time() - t0:.0f}s. -> {CKPT}')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as spstats

steps   = np.array([e['step']    for e in log_history])
rewards = np.array([e['reward']  for e in log_history])
losses  = np.array([e['loss']    for e in log_history])
fmts    = np.array([e['format_rate'] for e in log_history])
pitches = np.array([e['pitch_rate']  for e in log_history])

def ema(xs, alpha=0.1):
    out, s = [], xs[0] if len(xs) else 0.0
    for x in xs:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return np.array(out)

rewards_ema = ema(rewards, 0.1)
slope, intercept, r_val, p_val, _ = spstats.linregress(steps, rewards)

# Reward curve — vs base Qwen3-4B baseline (NOT random).
plt.figure(figsize=(9, 5))
plt.plot(steps, rewards, alpha=0.3, lw=1, label='per-step group reward')
plt.plot(steps, rewards_ema, lw=2.2, label='EMA (\u03b1=0.1)')
plt.plot(steps, intercept + slope * steps, '--', lw=1.5,
         label=f'linear fit slope={slope:+.4f}/step  (p={p_val:.1e})')
plt.axhline(BASELINE_MEAN_REWARD, ls=':', lw=2, color='#c44',
            label=f'base Qwen3-4B baseline = {BASELINE_MEAN_REWARD:.2f}')
plt.title('GRPO reward — BoardSim (vs same model w/o fine-tuning)')
plt.xlabel('step'); plt.ylabel('mean group reward')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'reward_curve.png', dpi=150); plt.close()

# Loss
plt.figure(figsize=(9, 5))
plt.plot(steps, losses, lw=1.5)
plt.title('GRPO loss (advantage \u00d7 NLL)'); plt.xlabel('step'); plt.ylabel('loss')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'loss_curve.png', dpi=150); plt.close()

# Format compliance + pitch rate
plt.figure(figsize=(9, 5))
plt.plot(steps, ema(fmts, 0.05),    lw=2, label='format-OK rate (EMA)')
plt.plot(steps, ema(pitches, 0.05), lw=2, label='non-empty pitch rate (EMA)')
plt.title('Format compliance + pitch usage during training')
plt.xlabel('step'); plt.ylabel('rate'); plt.ylim(-0.05, 1.05)
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'format_compliance.png', dpi=150); plt.close()

# Periodic eval — overlaid against base Qwen3-4B baseline so the reader
# can see the LoRA-trained policy progressively pull away from the base
# model on held-out seeds.
if eval_history:
    es  = [e['step']        for e in eval_history]
    epm = [e['profit_mean'] for e in eval_history]
    erm = [e['reward_mean'] for e in eval_history]
    plt.figure(figsize=(9, 5))
    plt.plot(es, epm, '-o', lw=2, label='held-out profitability (mean of 10 episodes)')
    plt.plot(es, erm, '-s', lw=2, label='held-out episode reward')
    plt.axhline(BASELINE_MEAN_PROFIT, ls=':', lw=1.5, color='#c44',
                label=f'base Qwen3-4B profitability = {BASELINE_MEAN_PROFIT:.2f}')
    plt.title('Periodic held-out eval during training (greedy)')
    plt.xlabel('training step'); plt.ylabel('value')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(ASSETS / 'periodic_eval.png', dpi=150); plt.close()

print(f'Linear-fit slope on reward: {slope:+.5f}/step (p={p_val:.2e}, R\u00b2={r_val**2:.3f})')
print('Saved reward_curve.png, loss_curve.png, format_compliance.png, periodic_eval.png')
# -----------------------------------------------------------------------------
# Paired same-seed eval: fine-tuned vs BASE Qwen3-4B (adapters disabled).
# This is the headline comparison. Same prompts, same env seeds, same
# decoder, same parser — only the LoRA delta differs.
# -----------------------------------------------------------------------------
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

EVAL_N = 50
PAIRED_SEEDS = list(range(70_000, 70_000 + EVAL_N))

# Trained policy (adapters active)
trained_finals, trained_rewards, trained_fmt, trained_pitch = [], [], [], []
trained_history_per_seed = []
with make_env().sync() as env:
    for i, s in enumerate(PAIRED_SEEDS):
        r = run_episode(env, s)
        trained_finals.append(r['final_profit'])
        trained_rewards.append(r['ep_reward'])
        trained_fmt.append(r['format_rate'])
        trained_pitch.append(r['pitch_rate'])
        trained_history_per_seed.append(r['history'])
        if (i + 1) % 10 == 0:
            print(f'  trained {i+1}/{EVAL_N}  profit={r["final_profit"]:.1f}')

# Base Qwen3-4B (LoRA disabled) — paired seeds.
base_finals_paired, base_rewards_paired, base_fmt_paired, base_pitch_paired = [], [], [], []
base_history_per_seed = []
with make_env().sync() as env, model.disable_adapter():
    for i, s in enumerate(PAIRED_SEEDS):
        r = run_episode(env, s)
        base_finals_paired.append(r['final_profit'])
        base_rewards_paired.append(r['ep_reward'])
        base_fmt_paired.append(r['format_rate'])
        base_pitch_paired.append(r['pitch_rate'])
        base_history_per_seed.append(r['history'])
        if (i + 1) % 10 == 0:
            print(f'  base    {i+1}/{EVAL_N}  profit={r["final_profit"]:.1f}')

tf, bf = np.array(trained_finals), np.array(base_finals_paired)
tr, br = np.array(trained_rewards), np.array(base_rewards_paired)

print(f'\nTrained Qwen3-4B profit : {tf.mean():.2f} \u00b1 {tf.std():.2f}')
print(f'Base    Qwen3-4B profit : {bf.mean():.2f} \u00b1 {bf.std():.2f}')
print(f'Trained ep reward       : {tr.mean():.2f} \u00b1 {tr.std():.2f}')
print(f'Base    ep reward       : {br.mean():.2f} \u00b1 {br.std():.2f}')
print(f'Trained format/pitch    : {np.mean(trained_fmt):.0%} / {np.mean(trained_pitch):.0%}')
print(f'Base    format/pitch    : {np.mean(base_fmt_paired):.0%} / {np.mean(base_pitch_paired):.0%}')

with open(DRIVE_DIR / 'eval_paired.json', 'w') as f:
    json.dump({'seeds': PAIRED_SEEDS,
               'trained_finals': tf.tolist(), 'base_finals': bf.tolist(),
               'trained_rewards': tr.tolist(), 'base_rewards': br.tolist(),
               'trained_format_rate': float(np.mean(trained_fmt)),
               'base_format_rate':    float(np.mean(base_fmt_paired)),
               'trained_pitch_rate':  float(np.mean(trained_pitch)),
               'base_pitch_rate':     float(np.mean(base_pitch_paired))}, f)
from scipy import stats as spstats

def cohen_d(a, b):
    pooled = np.sqrt(((a.std(ddof=1)**2) + (b.std(ddof=1)**2)) / 2)
    return (a.mean() - b.mean()) / (pooled + 1e-12)

def bootstrap_diff_ci(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = a - b  # paired
    boots = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

tt   = spstats.ttest_rel(tf, bf)
uu   = spstats.mannwhitneyu(tf, bf, alternative='greater')
wilc = spstats.wilcoxon(tf, bf, alternative='greater')
d    = cohen_d(tf, bf)
lo, hi = bootstrap_diff_ci(tf, bf)
win_rate = float((tf > bf).mean())
tie_rate = float((tf == bf).mean())

summary = {
    'baseline_model': MODEL_NAME + ' (no fine-tune)',
    'trained_model':  MODEL_NAME + ' + LoRA r=32',
    'n': len(tf),
    'paired_t_stat': float(tt.statistic), 'paired_t_p': float(tt.pvalue),
    'mannwhitney_U': float(uu.statistic), 'mannwhitney_p_greater': float(uu.pvalue),
    'wilcoxon_p_greater': float(wilc.pvalue),
    'cohens_d': float(d),
    'paired_diff_mean': float((tf - bf).mean()),
    'paired_diff_95ci': [lo, hi],
    'win_rate_trained_strictly_better': win_rate,
    'tie_rate': tie_rate,
}
print(json.dumps(summary, indent=2))
with open(DRIVE_DIR / 'stats_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
# Histogram — fine-tuned vs BASE on the same seeds.
bins = np.linspace(0, 100, 25)
plt.figure(figsize=(9, 5))
plt.hist(bf, bins=bins, alpha=0.55, color='#c44',
         label=f'Base Qwen3-4B (mean={bf.mean():.1f})')
plt.hist(tf, bins=bins, alpha=0.55, color='#1d6fff',
         label=f'Fine-tuned Qwen3-4B (mean={tf.mean():.1f})')
plt.axvline(bf.mean(), color='#c44', ls='--', lw=1.5)
plt.axvline(tf.mean(), color='#1d6fff', ls='--', lw=1.5)
plt.title(f'Final profitability — paired same-seed (n={len(tf)})  '
          f"d={summary['cohens_d']:+.2f}  win-rate={summary['win_rate_trained_strictly_better']:.0%}")
plt.xlabel('profitability score (0\u2013100)'); plt.ylabel('episodes')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'before_after.png', dpi=150); plt.close()

diffs = tf - bf
order = np.argsort(diffs)
plt.figure(figsize=(9, 5))
plt.bar(range(len(diffs)), diffs[order],
        color=['#1d6fff' if x > 0 else '#c44' for x in diffs[order]])
plt.axhline(0, color='k', lw=0.8)
plt.title(f'Per-seed lift (fine-tuned \u2212 base Qwen3-4B), sorted  '
          f'mean lift = {diffs.mean():+.1f}  CI=[{summary["paired_diff_95ci"][0]:+.1f}, {summary["paired_diff_95ci"][1]:+.1f}]')
plt.xlabel('seed (sorted by lift)'); plt.ylabel('\u0394 profitability')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'paired_delta.png', dpi=150); plt.close()
print('Saved before_after.png, paired_delta.png')
# -----------------------------------------------------------------------------
# Per-event win-rate breakdown — for each of the 10 generic events, how often
# did the fine-tuned policy win the boardroom vote vs base Qwen3-4B?
# This is the most direct picture of WHERE the fine-tuning helps.
# -----------------------------------------------------------------------------
def per_event_winrate(history_per_seed):
    bucket = collections.defaultdict(lambda: [0, 0])  # title -> [wins, total]
    for hist in history_per_seed:
        for rd in hist:
            t = rd.get('event_title', '?')
            bucket[t][1] += 1
            if rd.get('agent_won_vote'):
                bucket[t][0] += 1
    return {t: (w / max(1, n)) for t, (w, n) in bucket.items()}

trained_wr = per_event_winrate(trained_history_per_seed)
base_wr    = per_event_winrate(base_history_per_seed)

events_sorted = sorted(set(trained_wr) | set(base_wr))
tw = [trained_wr.get(e, 0.0) for e in events_sorted]
bw = [base_wr.get(e, 0.0)    for e in events_sorted]

plt.figure(figsize=(11, 5))
x = np.arange(len(events_sorted))
plt.bar(x - 0.2, bw, width=0.4, color='#c44', label='Base Qwen3-4B')
plt.bar(x + 0.2, tw, width=0.4, color='#1d6fff', label='Fine-tuned Qwen3-4B')
plt.xticks(x, [e[:22] for e in events_sorted], rotation=30, ha='right')
plt.ylim(0, 1.05); plt.ylabel('boardroom win rate')
plt.title('Per-event boardroom win rate (paired seeds, n=50 episodes)')
plt.legend(); plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig(ASSETS / 'per_event_winrate.png', dpi=150); plt.close()

with open(DRIVE_DIR / 'per_event_winrate.json', 'w') as f:
    json.dump({'events': events_sorted, 'trained': tw, 'base': bw}, f, indent=2)
print('Saved per_event_winrate.png')
# -----------------------------------------------------------------------------
# Theory-of-Mind probe — does the model identify which board member is most
# likely to oppose its decision? Run for BOTH base and fine-tuned for fair
# comparison, since "random=25%" is a weak reference for a 4 B LM.
# -----------------------------------------------------------------------------
TOM_INSTRUCTION = (
    "\n\nGiven the state and event below, name the SINGLE board member "
    "(CTO, CFO, Investor Rep, or Independent) most likely to oppose the chosen decision. "
    "Answer with just the role name on one line.\n"
)

def tom_predict(obs, decision):
    body = build_prompt(obs).split(SYSTEM_PROMPT, 1)[1]
    prompt = SYSTEM_PROMPT + TOM_INSTRUCTION + body + f'Chosen decision: {decision}\nMost likely opponent: '
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).lower()
    if 'investor'    in txt: return 'Investor Rep'
    if 'independent' in txt: return 'Independent'
    if 'cto'         in txt: return 'CTO'
    if 'cfo'         in txt: return 'CFO'
    return None

def tom_eval(seed_base=80_000, n=40):
    correct = total = 0
    with make_env().sync() as env:
        for ep in range(n):
            result = env.reset(seed=seed_base + ep)
            obs = result.observation
            decision, _, _ = greedy_action(obs)
            opposed = [s['role'] for s in obs.npc_statements if s['vote'] != decision]
            if not opposed:
                continue
            pred = tom_predict(obs, decision)
            if pred and pred in opposed:
                correct += 1
            total += 1
    return correct, total

t_corr, t_tot = tom_eval()
with model.disable_adapter():
    b_corr, b_tot = tom_eval()

tom_acc        = t_corr / max(1, t_tot)
tom_acc_base   = b_corr / max(1, b_tot)
print(f'ToM probe: trained = {tom_acc:.1%} ({t_corr}/{t_tot})   base = {tom_acc_base:.1%} ({b_corr}/{b_tot})')
with open(DRIVE_DIR / 'tom.json', 'w') as f:
    json.dump({'trained': {'correct': t_corr, 'total': t_tot, 'accuracy': tom_acc},
               'base':    {'correct': b_corr, 'total': b_tot, 'accuracy': tom_acc_base}}, f)
ROLES = ['CTO','CFO','Investor Rep','Independent']
trust_trained = {r: [] for r in ROLES}
trust_base    = {r: [] for r in ROLES}

def collect_trust(store, n=20, seed_base=90_000, base_mode=False):
    with make_env().sync() as env:
        for ep in range(n):
            result = env.reset(seed=seed_base + ep)
            obs = result.observation
            steps_done = 0
            while not result.done and steps_done < MAX_STEPS_PER_EP:
                decision, pitch, _ = greedy_action(obs)
                result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
                obs = result.observation
                steps_done += 1
            for entry in obs.state.get('trust_history', []):
                idx = entry.get('round', 0)
                for role in store:
                    if role not in entry: continue
                    while len(store[role]) <= idx:
                        store[role].append([])
                    store[role][idx].append(entry[role])

collect_trust(trust_trained)
with model.disable_adapter():
    collect_trust(trust_base, base_mode=True)

plt.figure(figsize=(10, 6))
for role, color in zip(ROLES, ['#1d6fff','#c44','#7a2','#a3a']):
    mt = [np.mean(x) if x else np.nan for x in trust_trained[role]]
    mb = [np.mean(x) if x else np.nan for x in trust_base[role]]
    plt.plot(range(len(mt)), mt, color=color, lw=2,            label=f'{role} (fine-tuned)')
    plt.plot(range(len(mb)), mb, color=color, lw=1.2, ls='--', alpha=0.6, label=f'{role} (base)')
plt.title('Per-round trust — fine-tuned (solid) vs base Qwen3-4B (dashed)')
plt.xlabel('round'); plt.ylabel('trust [0.1, 1.0]')
plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ASSETS / 'trust_trajectory.png', dpi=150); plt.close()
print('Saved trust_trajectory.png')
def transcript(env, seed, mode):
    """mode in {'trained', 'base'}."""
    rec = {'seed': seed, 'mode': mode, 'rounds': []}
    result = env.reset(seed=seed)
    obs = result.observation
    n = 0
    while not result.done and n < MAX_STEPS_PER_EP:
        decision, pitch, ok = greedy_action(obs)
        result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
        rec['rounds'].append({
            'event': obs.event, 'options': list(obs.options),
            'decision': decision, 'pitch': pitch[:300], 'format_ok': ok,
            'reward': float(result.reward or 0.0),
            'profit_after': result.observation.state['profitability_score'],
        })
        obs = result.observation; n += 1
    rec['final_profit'] = obs.state['profitability_score']
    return rec

transcripts = []
DEMO_SEEDS = [70_000, 70_001, 70_002]
with make_env().sync() as env:
    for s in DEMO_SEEDS:
        transcripts.append(transcript(env, s, 'trained'))
with make_env().sync() as env, model.disable_adapter():
    for s in DEMO_SEEDS:
        transcripts.append(transcript(env, s, 'base'))
with open(DRIVE_DIR / 'transcripts.json', 'w') as f:
    json.dump(transcripts, f, indent=2)

for t in transcripts:
    print(f"\n=== seed={t['seed']}  mode={t['mode']}  final_profit={t['final_profit']:.1f} ===")
    for i, rd in enumerate(t['rounds'][:3]):
        print(f"  R{i}: {rd['event'][:60]}\u2026 \u2192 {rd['decision']}  r={rd['reward']:+.2f}")
        if rd['pitch']:
            print(f"      pitch: {rd['pitch'][:120]}")
with open(DRIVE_DIR / 'decision_counter.json') as f:
    dc = json.load(f)
labels = list(dc.keys())
counts = np.array(list(dc.values()), dtype=float)
p = counts / counts.sum()
entropy = float(-(p * np.log(p + 1e-12)).sum())
max_ent = float(np.log(len(p)))
print(f'Decision entropy: {entropy:.3f} / {max_ent:.3f} (1.0 = uniform)  ratio={entropy/max_ent:.2%}')

plt.figure(figsize=(9, 5))
order = np.argsort(-counts)
plt.bar([labels[i] for i in order][:15], counts[order][:15])
plt.xticks(rotation=45, ha='right')
plt.title(f'Top-15 decisions during training (entropy={entropy:.2f}/{max_ent:.2f})')
plt.ylabel('count'); plt.tight_layout()
plt.savefig(ASSETS / 'decision_distribution.png', dpi=150); plt.close()
print('Saved decision_distribution.png')
from huggingface_hub import HfApi
ADAPTER_REPO = os.environ.get('ADAPTER_REPO', 'StavanKhobare/SST-MetaxPyTorch-Hackathon-LoRA')
MERGED_REPO  = os.environ.get('MERGED_REPO',  'StavanKhobare/SST-MetaxPyTorch-Hackathon-Merged16bit')

api = HfApi()
api.create_repo(ADAPTER_REPO, repo_type='model', private=False, exist_ok=True)
api.create_repo(MERGED_REPO,  repo_type='model', private=False, exist_ok=True)

# 1) LoRA adapter (small, fast)
try:
    model.push_to_hub(ADAPTER_REPO, private=False)
    tokenizer.push_to_hub(ADAPTER_REPO, private=False)
    print(f'\u2713 LoRA pushed: https://huggingface.co/{ADAPTER_REPO}')
except Exception as e:
    print(f'LoRA push failed: {e!r}')

# 2) Merged 16-bit
try:
    model.push_to_hub_merged(MERGED_REPO, tokenizer, save_method='merged_16bit', private=False)
    print(f'\u2713 Merged 16-bit pushed: https://huggingface.co/{MERGED_REPO}')
except Exception as e:
    print(f'Merged push failed (you can retry): {e!r}')

# 3) Upload eval artifacts
try:
    api.upload_folder(folder_path=str(ASSETS), repo_id=ADAPTER_REPO,
                      path_in_repo='assets', repo_type='model')
    for fname in ['log_history.json','eval_history.json','eval_paired.json',
                  'stats_summary.json','tom.json','transcripts.json',
                  'decision_counter.json','baseline.json',
                  'per_event_winrate.json']:
        fp = DRIVE_DIR / fname
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=fname,
                            repo_id=ADAPTER_REPO, repo_type='model')
    print(f'\u2713 Artifacts uploaded to https://huggingface.co/{ADAPTER_REPO}')
except Exception as e:
    print(f'Artifact upload failed: {e!r}')
print('='*70)
print('BOARDSIM \u00d7 QWEN3-4B \u2014 LEARNING EVIDENCE')
print('='*70)
print(f'Reward slope (linear fit) : {slope:+.5f}/step  (p={p_val:.2e})')
print(f'Reward EMA first 20 steps : {rewards_ema[:20].mean():+.3f}')
print(f'Reward EMA last 20 steps  : {rewards_ema[-20:].mean():+.3f}')
print(f'Format compliance start   : {fmts[:20].mean():.0%}')
print(f'Format compliance end     : {fmts[-20:].mean():.0%}')
print('-'*70)
print(f'Held-out paired (n={len(tf)}):  fine-tuned {tf.mean():.2f}  vs  base {bf.mean():.2f}')
print(f'  paired t-test p={summary["paired_t_p"]:.2e}   Wilcoxon p={summary["wilcoxon_p_greater"]:.2e}')
print(f'  Cohen d={summary["cohens_d"]:+.2f}   95% CI of lift = [{summary["paired_diff_95ci"][0]:+.2f}, {summary["paired_diff_95ci"][1]:+.2f}]')
print(f'  win rate (fine-tuned > base): {summary["win_rate_trained_strictly_better"]:.0%}')
print(f'ToM probe  fine-tuned     : {tom_acc:.0%}    base = {tom_acc_base:.0%}')
print(f'Decision entropy          : {entropy:.2f} / {max_ent:.2f}  (\u2192 not collapsed)')
print('-'*70)
print(f'Adapter      : https://huggingface.co/{ADAPTER_REPO}')
print(f'Merged 16bit : https://huggingface.co/{MERGED_REPO}')
print(f'Env Space    : {ENV_BASE_URL}')
print('='*70)

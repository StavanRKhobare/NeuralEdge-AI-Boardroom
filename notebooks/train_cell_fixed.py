# =============================================================================
# GRPO training cell — fixed version
#
# Fixes:
#  1. RuntimeError "variable modified by an inplace operation" on loss.backward().
#     Root cause: model.generate() leaves use_cache=True, and the subsequent
#     forward pass returns logits that share storage with KV-cache buffers,
#     which get mutated later. Fix: force use_cache=False on the training
#     forward pass, and .clone() the logits slice before computing log_softmax.
#
#  2. GPU OOM on cell re-run. Root cause: re-running the cell creates a fresh
#     AdamW (which holds momentum buffers ~= model size) without freeing the
#     previous one. Fix: explicit cleanup of any prior optimizer / cached
#     tensors at the top of the cell + gc + empty_cache. Model itself is NOT
#     reloaded here (load it once in an earlier cell); we just reuse it.
#
#  3. wandb deprecation warning for reinit=True. Use finish_previous=True only.
# =============================================================================

import os, gc, json, time, collections
import torch
from torch.optim import AdamW

# ---- 0. cleanup any leftover state from previous runs of this cell ----------
for _name in ('optimizer', 'gen_out', 'out', 'logits', 'loss',
              'log_probs', 'token_nll', 'per_seq_nll', 'advantages'):
    if _name in globals():
        try:
            del globals()[_name]
        except Exception:
            pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---- 1. config --------------------------------------------------------------
NUM_STEPS  = int(os.environ.get('NUM_STEPS', 100))
GROUP_SIZE = int(os.environ.get('GROUP_SIZE', 4))
LR         = 5e-6
GRAD_CLIP  = 1.0
TEMPERATURE, TOP_P = 1.0, 0.95
SAVE_EVERY = 25
EVAL_AT    = {0, 25, 50, 75, NUM_STEPS - 1}

# Critical: kill KV cache on the training forward pass.
# generate() will still build its own cache internally; we override afterwards.
model.config.use_cache = False
model.gradient_checkpointing_disable() if hasattr(model, 'gradient_checkpointing_disable') else None
model.train()

# ---- 2. wandb (no deprecated reinit) ----------------------------------------
WANDB_OK = False
if os.environ.get('WANDB_API_KEY'):
    try:
        import wandb
        wandb.init(
            project='boardsim-qwen3-grpo',
            name='boardsim-qwen3-1p7b-kaggle',
            config={'num_steps': NUM_STEPS, 'group_size': GROUP_SIZE, 'lr': LR,
                    'temperature': TEMPERATURE, 'top_p': TOP_P, 'model': MODEL_NAME},
            finish_previous=True,
        )
        WANDB_OK = True
    except Exception as e:
        print(f'WARN: wandb.init failed: {e}')

# ---- 3. optimizer (single owner, freshly built each cell run) ---------------
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
)

log_history, eval_history = [], []
decision_counter = collections.Counter()
t0 = time.time()

# ---- 4. training loop -------------------------------------------------------
with make_env().sync() as env_train, \
     make_env().sync() as env_score, \
     make_env().sync() as env_eval:

    for step in range(NUM_STEPS):
        # 4a. rollout
        result = env_train.reset(seed=step)
        obs = result.observation
        prompt = build_prompt(obs)
        enc = tokenizer(prompt, return_tensors='pt',
                        truncation=True, max_length=1024).to(device)
        prompt_len = enc.input_ids.shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # cache OK during generate (no_grad context)
            )
        # Detach + clone so no autograd ties to generate's internal buffers.
        gen_out = gen_out.detach().clone()

        # 4b. score each completion
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
        advantages = advantages.detach()

        # 4c. policy update — fresh forward, NO cache, clone logits
        optimizer.zero_grad(set_to_none=True)

        full_ids = gen_out
        attn     = (full_ids != tokenizer.pad_token_id).long()
        loss_mask = attn.clone()
        loss_mask[:, :prompt_len] = 0

        out = model(
            input_ids=full_ids,
            attention_mask=attn,
            use_cache=False,         # <-- key fix
            return_dict=True,
        )
        # Clone the slice so backward sees a tensor whose storage we own.
        logits  = out.logits[:, :-1, :].float().clone()
        targets = full_ids[:, 1:].contiguous()
        mask    = loss_mask[:, 1:].float()

        log_probs   = torch.nn.functional.log_softmax(logits, dim=-1)
        token_nll   = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        per_seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = (advantages * per_seq_nll).mean()

        loss.backward()
        total_loss_val = float(loss.detach().item())

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()

        # Free per-step graph tensors before next iter (helps on tight VRAM).
        del out, logits, log_probs, token_nll, per_seq_nll, loss

        # 4d. log
        rec = {
            'step':        step,
            'reward':      float(rewards_t.mean().item()),
            'reward_std':  float(rewards_t.std().item()) if rewards_t.numel() > 1 else 0.0,
            'reward_max':  float(rewards_t.max().item()),
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

        # 4e. periodic eval
        if step in EVAL_AT:
            ev = periodic_eval(env_eval)
            ev['step'] = step
            eval_history.append(ev)
            print(f"  [eval@{step}] profit={ev['profit_mean']:.2f}  "
                  f"reward={ev['reward_mean']:.2f}  fmt={ev['format_rate']:.0%}")
            if WANDB_OK:
                wandb.log({f'eval/{k}': v for k, v in ev.items() if k != 'step'}, step=step)

        # 4f. checkpoint
        if step > 0 and step % SAVE_EVERY == 0:
            model.save_pretrained(str(CKPT))
            tokenizer.save_pretrained(str(CKPT))
            with open(WORK_DIR / 'log_history.json', 'w') as f:
                json.dump(log_history, f)
            with open(WORK_DIR / 'eval_history.json', 'w') as f:
                json.dump(eval_history, f)

# ---- 5. final save ----------------------------------------------------------
model.save_pretrained(str(CKPT))
tokenizer.save_pretrained(str(CKPT))
with open(WORK_DIR / 'log_history.json', 'w') as f:
    json.dump(log_history, f)
with open(WORK_DIR / 'eval_history.json', 'w') as f:
    json.dump(eval_history, f)
with open(WORK_DIR / 'decision_counter.json', 'w') as f:
    json.dump(dict(decision_counter), f)
if WANDB_OK:
    wandb.finish()
print(f'Training done. {len(log_history)} steps in {time.time() - t0:.0f}s. -> {CKPT}')

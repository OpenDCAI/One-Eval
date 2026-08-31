#!/usr/bin/env python
"""
τ2-bench evaluation bridge script for One-Eval.
Bridges One-Eval's env-var conventions to tau2-bench's programmatic API.

Flow:
  1. Read model config from One-Eval env vars
  2. Configure LiteLLM environment for OpenAI-compatible endpoint
  3. Invoke tau2's run_domain() for each domain
  4. Collect pass_rate into a standardized scores JSON

Model config is read from environment variables:
  OPENAI_API_KEY        - API key for LLM access
  OPENAI_API_BASE       - Base URL for OpenAI-compatible endpoint
  ONEEVAL_MODEL_NAME    - Model name (will be prefixed with "openai/")
  ONEEVAL_USER_MODEL    - User simulator model (optional, defaults to agent model)
  ONEEVAL_MAX_SAMPLES   - Max number of tasks to evaluate per domain (-1 = all)
  ONEEVAL_TAU2_DOMAINS  - Comma-separated domains to evaluate (default: airline,retail,telecom)
  ONEEVAL_TAU2_TRIALS   - Number of trials per task (default: 4)
  ONEEVAL_AGENT_REASONING_EFFORT - reasoning_effort for agent LLM (e.g. none, low, medium, high)
  ONEEVAL_USER_REASONING_EFFORT  - reasoning_effort for user simulator LLM (e.g. low)
  ONEEVAL_EVAL_MODEL    - NL evaluator model (default: gpt-4.1)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tau2_bench_oneeval")

DEFAULT_DOMAINS = "airline,retail,telecom"


def parse_args():
    parser = argparse.ArgumentParser(description="τ2-bench evaluation for One-Eval")
    parser.add_argument("--domains", type=str, default=DEFAULT_DOMAINS,
                        help="Comma-separated domains to evaluate")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--user_model", type=str, default="",
                        help="User simulator model (defaults to agent model)")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_trials", type=int, default=4,
                        help="Number of evaluation trials per task")
    parser.add_argument("--task_seed", type=int, default=300,
                        help="Random seed for task sampling/ordering")
    parser.add_argument("--agent_reasoning_effort", type=str, default="none",
                        help="Reasoning effort for agent LLM (none/low/medium/high)")
    parser.add_argument("--user_reasoning_effort", type=str, default="low",
                        help="Reasoning effort for user simulator LLM")
    parser.add_argument("--eval_model", type=str, default="",
                        help="NL evaluator model (defaults to agent model)")
    parser.add_argument("--max_concurrency", type=int, default=5)
    return parser.parse_args()


def setup_env():
    """Verify environment for LiteLLM's OpenAI provider.

    API credentials are passed via environment variables (OPENAI_API_KEY,
    OPENAI_API_BASE) set by the orchestrator.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set in environment")


def build_llm_string(model_name: str) -> str:
    """Build LiteLLM model string. Prefix with 'openai/' if not already prefixed."""
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _patch_tau2_config(agent_model: str, user_model: str, eval_model: str):
    """Patch tau2.config defaults BEFORE other tau2 modules import them.

    Only override agent and user simulator models.
    NL evaluator (nl_assertions, env_interface) uses eval_model if provided,
    otherwise defaults to agent_model.
    """
    import importlib.util
    config_path = Path("src/tau2/config.py")
    if not config_path.exists():
        return
    spec = importlib.util.spec_from_file_location("tau2.config", str(config_path))
    config_mod = importlib.util.module_from_spec(spec)
    sys.modules["tau2.config"] = config_mod
    spec.loader.exec_module(config_mod)
    config_mod.DEFAULT_LLM_AGENT = agent_model
    config_mod.DEFAULT_LLM_USER = user_model
    config_mod.DEFAULT_LLM_EVAL_USER_SIMULATOR = user_model

    eval_llm = build_llm_string(eval_model)
    config_mod.DEFAULT_LLM_NL_ASSERTIONS = eval_llm
    config_mod.DEFAULT_LLM_ENV_INTERFACE = eval_llm


def run_tau2_evaluation_api(args, domain: str, num_tasks: int = -1):
    """Run τ2-bench evaluation for a single domain using the programmatic API.

    Args:
        num_tasks: Number of tasks to evaluate in this domain.
                   -1 means all available tasks.
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    agent_llm = build_llm_string(args.model_name)
    user_llm = build_llm_string(args.user_model) if args.user_model else agent_llm

    # Build llm_args with reasoning_effort if specified
    llm_args_agent = {"temperature": 0.0}
    llm_args_user = {"temperature": 0.0}

    if args.agent_reasoning_effort and args.agent_reasoning_effort != "none":
        llm_args_agent["reasoning_effort"] = args.agent_reasoning_effort
    if args.user_reasoning_effort:
        llm_args_user["reasoning_effort"] = args.user_reasoning_effort

    config_kwargs = dict(
        domain=domain,
        agent="llm_agent",
        llm_agent=agent_llm,
        llm_user=user_llm,
        llm_args_agent=llm_args_agent,
        llm_args_user=llm_args_user,
        num_trials=args.num_trials,
        seed=args.task_seed,
        max_concurrency=args.max_concurrency,
    )
    if num_tasks > 0:
        config_kwargs["num_tasks"] = num_tasks

    run_config = TextRunConfig(**config_kwargs)

    log.info(f"Starting τ2-bench evaluation (API) for domain: {domain}")
    results = run_domain(run_config)
    return results


def run_tau2_evaluation_cli(args, domain: str, num_tasks: int = -1):
    """Fallback: run τ2-bench via CLI subprocess for a single domain."""
    import subprocess

    agent_llm = build_llm_string(args.model_name)
    user_llm = build_llm_string(args.user_model) if args.user_model else agent_llm

    cmd = [
        sys.executable, "-m", "tau2", "run",
        "--domain", domain,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--num-trials", str(args.num_trials),
    ]
    if num_tasks > 0:
        cmd.extend(["--num-tasks", str(num_tasks)])

    log.info(f"Running CLI: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"CLI failed (exit {result.returncode}):\n{result.stderr}")
        return None

    log.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return parse_cli_results()


def parse_cli_results():
    """Parse results from tau2's data/simulations/ directory after CLI run."""
    sim_dir = Path("data/simulations")
    if not sim_dir.exists():
        log.warning("data/simulations/ not found")
        return None

    result_files = sorted(
        sim_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for f in result_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "reward" in str(data):
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def run_single_domain(args, domain: str, num_tasks: int = -1):
    """Run evaluation for one domain, with CLI fallback on API failure."""
    try:
        results = run_tau2_evaluation_api(args, domain, num_tasks)
        if results:
            return results
    except (TypeError, ImportError) as e:
        log.warning(f"Programmatic API failed for {domain} ({e}), falling back to CLI...")

    return run_tau2_evaluation_cli(args, domain, num_tasks)


def compute_domain_scores(results) -> dict:
    """Compute Pass^k metrics for a single domain's results using tau2's official API."""
    if isinstance(results, dict):
        if "pass_rate" in results:
            return results
        rewards = [v for v in results.values() if isinstance(v, (int, float))]
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "pass_rate": pass_rate,
            "pass_hat_ks": {1: pass_rate},
            "num_tasks": len(rewards),
            "num_passed": sum(1 for r in rewards if r >= 1.0),
        }

    # Use official compute_metrics for proper Pass^k calculation
    pass_hat_ks = {}
    try:
        from tau2.metrics.agent_metrics import compute_metrics
        metrics = compute_metrics(results)
        pass_hat_ks = dict(metrics.pass_hat_ks) if metrics.pass_hat_ks else {}
    except (ImportError, Exception) as e:
        log.warning(f"compute_metrics failed ({e}), falling back to manual calculation")

    simulations = results.simulations if hasattr(results, "simulations") else results

    rewards = []
    details = []

    for r in simulations:
        reward = r.reward_info.reward if r.reward_info else 0.0
        rewards.append(reward)

        messages = []
        if hasattr(r, 'messages') and r.messages:
            for msg in r.messages:
                msg_dict = {}
                if hasattr(msg, 'role'):
                    msg_dict["role"] = msg.role
                if hasattr(msg, 'content'):
                    content = str(msg.content)
                    msg_dict["content"] = content[:2000] if len(content) > 2000 else content
                    msg_dict["content_truncated"] = len(content) > 2000
                if msg_dict:
                    messages.append(msg_dict)

        goal = ""
        if hasattr(r, 'task'):
            if hasattr(r.task, 'goal') and r.task.goal:
                goal = str(r.task.goal)
            elif hasattr(r.task, 'description') and r.task.description:
                goal = str(r.task.description)
            elif hasattr(r.task, 'instruction') and r.task.instruction:
                goal = str(r.task.instruction)
            elif hasattr(r.task, '__dict__'):
                task_dict = r.task.__dict__
                for key in ['goal', 'description', 'instruction', 'prompt', 'text']:
                    if key in task_dict and task_dict[key]:
                        goal = str(task_dict[key])
                        break

        details.append({
            "task_id": str(r.task_id),
            "reward": float(reward),
            "num_messages": len(messages),
            "goal": goal,
            "messages": messages,
        })

    # Fallback if compute_metrics didn't work
    if not pass_hat_ks:
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
        pass_hat_ks = {1: pass_rate}

    pass_rate = pass_hat_ks.get(1, 0.0)
    num_passed = sum(1 for r in rewards if r >= 1.0)

    return {
        "pass_rate": pass_rate,
        "pass_hat_ks": pass_hat_ks,
        "num_tasks": len(set(d["task_id"] for d in details)),
        "num_passed": num_passed,
        "details": details,
    }


def write_oneeval_scores(args, domain_scores: dict, domains: list):
    """Write results in One-Eval's expected format."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Build per-sample JSONL
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name

    total_samples = 0
    with open(detail_path, "w", encoding="utf-8") as fout:
        for domain in domains:
            if domain not in domain_scores:
                continue

            details = domain_scores[domain].get("details", [])
            for item in details:
                task_id = item.get("task_id", "")
                goal = item.get("goal", "")
                reward = item.get("reward", 0.0)
                messages = item.get("messages", [])

                # Build prompt: use first user message as task description
                # tau2 tasks don't have predefined goals; the task is defined by user's first request
                prompt_text = f"[{domain}] Task {task_id}"
                if messages:
                    # Find first user message as the task description
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                            if user_msg and len(user_msg) > 10:  # Skip very short messages
                                prompt_text = f"[{domain}] {user_msg[:200]}"
                                break
                elif goal:
                    prompt_text = f"[{domain}] {goal}"

                # Format conversation as solution
                solution_lines = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    solution_lines.append(f"{role}: {content}")
                solution = "\n".join(solution_lines) if solution_lines else "(no conversation)"

                record = {
                    "task_id": f"{domain}_{task_id}",
                    "domain": domain,
                    "prompt": prompt_text,
                    "solution": solution,
                    "ground_truth": "",  # tau2 doesn't have explicit ground truth
                    "eval_score": reward,  # 0-1 reward
                    "eval_valid": True,
                    "num_messages": len(messages),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    # Compute overall Pass^k as equal-weight average across domains
    evaluated_domains = [d for d in domains if d in domain_scores]
    num_domains = len(evaluated_domains)

    overall_pass_hat_ks = {}
    if num_domains > 0:
        all_ks = set()
        for d in evaluated_domains:
            all_ks.update(domain_scores[d].get("pass_hat_ks", {}).keys())
        for k in sorted(all_ks):
            vals = [domain_scores[d]["pass_hat_ks"].get(k, 0.0) for d in evaluated_domains]
            overall_pass_hat_ks[k] = sum(vals) / num_domains

    pass1 = overall_pass_hat_ks.get(1, 0.0)

    # Reproduction metadata
    result = {
        "pass_rate": pass1,
        "pass_hat_ks": overall_pass_hat_ks,
        "average": {"pass_rate": pass1},
        "total_samples": total_samples,
        "detail_path": detail_name,
        "bench_name": "tau2_bench",
        "metadata": {
            "agent_model": args.model_name,
            "agent_reasoning_effort": args.agent_reasoning_effort or None,
            "user_model": args.user_model or args.model_name,
            "user_reasoning_effort": args.user_reasoning_effort or None,
            "eval_model": args.eval_model or args.model_name,
            "num_trials": args.num_trials,
            "task_seed": args.task_seed,
            "domains": evaluated_domains,
            "task_split": "base",
            "benchmark": "tau2-bench",
            "benchmark_ref": "v1.0.1",
        },
        "domains": evaluated_domains,
        "num_trials": args.num_trials,
        "task_seed": args.task_seed,
        "timestamp": timestamp,
    }

    # Add per-domain stats with full Pass^k
    for d in evaluated_domains:
        s = domain_scores[d]
        result[f"{d}_pass_rate"] = s["pass_rate"]
        result[f"{d}_pass_hat_ks"] = s.get("pass_hat_ks", {})
        result[f"{d}_num_tasks"] = s["num_tasks"]

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Scores written to: {score_file}")
    log.info(f"Overall Pass^1: {pass1:.4f}")
    if overall_pass_hat_ks:
        for k, v in sorted(overall_pass_hat_ks.items()):
            log.info(f"  Pass^{k}: {v:.4f}")
    for d in evaluated_domains:
        s = domain_scores[d]
        ks = s.get("pass_hat_ks", {})
        ks_str = ", ".join(f"Pass^{k}={v:.4f}" for k, v in sorted(ks.items()))
        log.info(f"  {d}: {ks_str} (tasks={s['num_tasks']})")

    return result


def main():
    args = parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    log.info("τ2-bench One-Eval bridge starting")
    log.info(f"  Domains: {domains}")
    log.info(f"  Agent model: {args.model_name}")
    log.info(f"  User model: {args.user_model or '(same as agent)'}")
    log.info(f"  num_trials={args.num_trials}, task_seed={args.task_seed}, max_samples={args.max_samples}")
    log.info(f"  Agent reasoning_effort: {args.agent_reasoning_effort or '(default)'}")
    log.info(f"  User reasoning_effort: {args.user_reasoning_effort or '(default)'}")
    log.info(f"  Eval model: {args.eval_model or '(same as agent)'}")

    # Clean stale simulation data directory
    sim_dir = Path("data/simulations")
    if sim_dir.exists():
        import shutil
        shutil.rmtree(sim_dir)
        log.info(f"Removed stale simulation directory: {sim_dir}")

    setup_env()

    agent_llm = build_llm_string(args.model_name)
    user_llm = build_llm_string(args.user_model) if args.user_model else agent_llm
    eval_llm_model = args.eval_model if args.eval_model else args.model_name
    _patch_tau2_config(agent_llm, user_llm, eval_llm_model)

    domain_scores = {}
    remaining = args.max_samples if args.max_samples > 0 else -1

    for domain in domains:
        if remaining == 0:
            break

        log.info("=" * 60)
        log.info(f"Evaluating domain: {domain}")
        log.info("=" * 60)

        results = run_single_domain(args, domain, remaining)

        if not results:
            log.error(f"No results for domain: {domain}")
            continue

        scores = compute_domain_scores(results)
        domain_scores[domain] = scores
        log.info(f"  {domain} pass_rate: {scores['pass_rate']:.4f}")

        if remaining > 0:
            remaining -= scores["num_tasks"]
            if remaining < 0:
                remaining = 0

    if not domain_scores:
        log.error("No evaluation results for any domain!")
        sys.exit(1)

    write_oneeval_scores(args, domain_scores, domains)
    log.info("Done!")


if __name__ == "__main__":
    main()

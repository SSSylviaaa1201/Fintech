"""SHAP-based explainer for DQN trading decisions.

Makes the black-box RL agent interpretable — shows which features drove each trade decision.
"""

import logging

import numpy as np
import pandas as pd
import shap
import torch

from rl_engine.dqn import DQNAgent
from rl_engine.env import FinancialTradingEnv, STATE_DIM

logger = logging.getLogger(__name__)

STATE_FEATURE_NAMES = [
    "price_ratio", "MA50_ratio", "MA200_ratio", "RSI_norm",
    "MACD_ratio", "position_pct", "cash_pct", "sentiment_score",
    "sentiment_ma5", "sentiment_trend", "sentiment_vol", "sentiment_mask",
]


class TradingExplainer:
    """Use SHAP DeepExplainer to explain DQN Q-value predictions."""

    def __init__(self, agent: DQNAgent):
        self.agent = agent
        self.explainer = None

    def fit(self, env: FinancialTradingEnv, n_background: int = 100):
        """Prepare SHAP explainer with background states sampled from env."""
        states = []
        for _ in range(n_background):
            state, _ = env.reset()
            steps_in_ep = min(len(env.df) - 1, np.random.randint(10, 50))
            for _ in range(steps_in_ep):
                action = env.action_space.sample()
                state, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
            states.append(state)

        background = np.array(states)
        self.explainer = shap.DeepExplainer(self.agent.q_network, torch.FloatTensor(background))
        logger.info("SHAP explainer fitted with %d background samples", n_background)

    def explain_state(self, state: np.ndarray) -> dict:
        """Explain why DQN chose its action for a given state."""
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_state()")

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        shap_values = self.explainer.shap_values(state_tensor)

        # shap_values is a list of arrays, one per action
        chosen_action = int(self.agent.select_action(state, evaluate=True))
        sv = shap_values[chosen_action].flatten()

        importance = sorted(
            zip(STATE_FEATURE_NAMES, sv),
            key=lambda x: abs(x[1]), reverse=True,
        )

        pos_drivers = [(f, round(v, 4)) for f, v in importance if v > 0.01]
        neg_drivers = [(f, round(v, 4)) for f, v in importance if v < -0.01]

        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        explanation_parts = [f"DQN chose **{action_names[chosen_action]}** because:"]

        if pos_drivers:
            drivers_str = ", ".join([f"{f} (+{v:.3f})" for f, v in pos_drivers[:3]])
            explanation_parts.append(f"- Positive signals: {drivers_str}")
        if neg_drivers:
            drivers_str = ", ".join([f"{f} ({v:.3f})" for f, v in neg_drivers[:3]])
            explanation_parts.append(f"- Negative signals: {drivers_str}")
        if not pos_drivers and not neg_drivers:
            explanation_parts.append("- All features near neutral (weak signal)")

        return {
            "shap_values": sv.tolist(),
            "feature_importance": [(f, round(v, 4)) for f, v in importance],
            "chosen_action": chosen_action,
            "action_name": action_names[chosen_action],
            "explanation_text": "\n".join(explanation_parts),
        }

    def global_feature_importance(self, df: pd.DataFrame, n_samples: int = 50) -> pd.Series:
        """Aggregate SHAP values across many states for global importance ranking."""
        env = FinancialTradingEnv(df)
        self.fit(env)

        records = []
        for _ in range(n_samples):
            state, _ = env.reset()
            for _ in range(np.random.randint(5, min(len(df) - env.current_step, 80))):
                try:
                    exp = self.explain_state(state)
                    record = {f: v for f, v in exp["feature_importance"]}
                    record["action"] = exp["chosen_action"]
                    records.append(record)
                except Exception:
                    break
                action = env.action_space.sample()
                state, _, term, _, _ = env.step(action)
                if term:
                    break

        report_df = pd.DataFrame(records)
        feat_cols = [c for c in report_df.columns if c in STATE_FEATURE_NAMES]
        if feat_cols:
            return report_df[feat_cols].abs().mean().sort_values(ascending=False)
        return pd.Series(dtype=float)

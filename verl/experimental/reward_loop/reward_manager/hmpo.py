# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HMPO (Hybrid Median-length Policy Optimization for Chain-of-Thought Compression) Reward Manager.

Core idea:
- HMPO uses the **median length of correct outputs** (score > 0) per UID group as the
  baseline b in the cosine-based length reward.
  If no correct output exists for a UID group, falls back to overlong_buffer_cfg.len.
"""

import asyncio
import inspect
import logging

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)


@register("hmpo")
class HMPORewardManager(RewardManagerBase):
    """HMPO (Hybrid Median-length Policy Optimization for Chain-of-Thought Compression) Reward Manager.

    Uses the median length of correct rollout outputs as the baseline b
    in the cosine-based length reward, instead of a fixed config value.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        overlong_buffer_cfg = config.reward.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def _compute_length_reward(self, n, b, L_max, beta, lambda_penalty,clip_reward_high):
        """Cosine-based length reward.

        b is the caller-provided baseline (median of correct outputs).

        Args:
            n: Current sequence length.
            b: Baseline length (median of correct outputs for this UID group).
            L_max: Maximum length threshold.
            beta: Cosine reward scaling coefficient.
            lambda_penalty: Length difference penalty coefficient.

        Returns:
            float: Length-related reward value, clipped to [0.0, 1.0].
        """
        
        length_reward = 0
        if n > b:
            return 0.0
        elif n <= b:
            # beta defaults to 1, lambda_penalty defaults to 0.6
            length_reward = beta * np.cos(np.pi * n / (2 * b)) + lambda_penalty
            # length_reward = beta *((b-n)/b) + lambda_penalty

        logger.warning(f"\nHMPO DEBUG: n:{n}, b:{b}, beta:{beta}, lambda_reward:{lambda_penalty}，length_reward：{length_reward}")
        return float(np.clip(length_reward, 0.0, clip_reward_high))


    def sync_run_batch(self, data: DataProto) -> list[dict]:
        """Synchronous parallel version of run_batch for local execution.

        Used by ray_trainer._compute_reward_hmpo_local() which calls sync_run_batch
        to avoid asyncio conflicts in the training loop.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        n_items = len(data)

        # 1. Prepare data
        all_responses_ids = [data[i].batch["responses"].tolist() for i in range(n_items)]
        all_attention_masks = [data[i].batch["attention_mask"].tolist() for i in range(n_items)]
        all_data_sources = [data[i].non_tensor_batch["data_source"] for i in range(n_items)]
        all_ground_truths = [data[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(n_items)]
        all_extra_infos = [data[i].non_tensor_batch.get("extra_info", {}) for i in range(n_items)]
        all_uids = [data[i].non_tensor_batch["uid"] for i in range(n_items)]

        # 2. Parallel decode + score
        def process_one_item(idx):
            resp_ids = np.array(all_responses_ids[idx])
            attn_mask = np.array(all_attention_masks[idx])
            response_length = resp_ids.shape[-1]
            valid_len = int(attn_mask[-response_length:].sum())

            resp_str = self.tokenizer.decode(resp_ids[:valid_len], skip_special_tokens=True)

            extra_reward_kwargs = (
                {
                    "reward_router_address": self.reward_router_address,
                    "reward_model_tokenizer": self.reward_model_tokenizer,
                }
                if self.reward_router_address is not None
                else {}
            )

            result = self.compute_score(
                data_source=all_data_sources[idx],
                solution_str=resp_str,
                ground_truth=all_ground_truths[idx],
                extra_info=all_extra_infos[idx],
                **extra_reward_kwargs,
            )

            return {
                "idx": idx,
                "score": result["score"] if isinstance(result, dict) else result,
                "extra": result if isinstance(result, dict) else {"acc": result},
                "valid_len": valid_len,
            }

        max_workers = min(32, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_one_item, range(n_items)))

        # 3. HMPO: compute median of CORRECT outputs per UID group
        valid_lengths_np = np.array([r["valid_len"] for r in batch_results])
        scores_np = np.array([r["score"] for r in batch_results])
        uids_np = np.array(all_uids)

        unique_uids, inverse_indices = np.unique(uids_np, return_inverse=True)
        num_prompts = len(unique_uids)

        per_prompt_medians = np.full(num_prompts, np.nan)
        for idx in range(num_prompts):
            mask = (inverse_indices == idx)
            group_lengths = valid_lengths_np[mask]
            group_scores = scores_np[mask]

            correct_mask = group_scores > 0
            correct_lengths = group_lengths[correct_mask]

            if len(correct_lengths) > 0:
                per_prompt_medians[idx] = np.median(correct_lengths)

        # logger.warning(f"HMPO Sync: Processed {n_items} items for {num_prompts} unique prompts.")

        # 4. Apply length reward
        final_results = []
        for i in range(n_items):
            res = batch_results[i]
            score = res["score"]
            reward_extra_info = res["extra"]

            if score == -1:
                score = 0
            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                L_max = self.overlong_buffer_cfg.get("L_max", self.max_resp_len)
                beta = self.overlong_buffer_cfg.get("beta", 1.0)
                lambda_penalty = self.overlong_buffer_cfg.get("lambda", 0.005)
                multi = self.overlong_buffer_cfg.get("multi", False)
                clip_reward_high = self.overlong_buffer_cfg.get("clip_reward_high", 1)
                fixed = self.overlong_buffer_cfg.get("fixed", False)
                targetlen = self.overlong_buffer_cfg.get("targetlen", 10000)

                n_len = res["valid_len"]
                b = per_prompt_medians[inverse_indices[i]]

                if fixed:
                    logger.warning(f"\n{'='*20} fixed {'='*20}\n")
                    b = targetlen
                    overlong_reward = 1 if n_len < b and score==1 else 0
                else:
                    

                    if np.isnan(b):
                        overlong_reward = 0.0
                    else:
                        overlong_reward = self._compute_length_reward(n_len, b, L_max, beta, lambda_penalty,clip_reward_high)

                
                if multi:    
                    reward = overlong_reward * score
                    logger.warning(f"\n{'='*20} multi {'='*20}\n")
                else:
                    reward += overlong_reward * score
                    logger.warning(f"\n{'='*20} add {'='*20}\n")
                if i == np.where(inverse_indices == inverse_indices[i])[0][0]:
                    group_mask = (inverse_indices == inverse_indices[i])
                    group_scores = scores_np[group_mask]
                    n_correct = int((group_scores > 0).sum())
                    logger.warning(
                        f"\n{'='*20} HMPO Sync UID Group DEBUG (UID: {all_uids[i]}) {'='*20}\n"
                        f"Group Rollout Lengths: {valid_lengths_np[group_mask].tolist()}\n"
                        f"Group Scores: {group_scores.tolist()}\n"
                        f"Correct outputs: {n_correct}/{len(group_scores)}\n"
                        f"Median-of-correct b: {'NaN (skip penalty)' if np.isnan(b) else f'{b:.2f}'}\n"
                        f"n:{n_len}\n"
                        f"compress rate based on b: {'NaN (skip compress)' if np.isnan(b) else 1-(n_len/b)}\n"
                        f"Beta: {beta}, Lambda: {lambda_penalty}\n"
                        f"overlong_reward: {overlong_reward}, acc_reward: {score},final_reward: {reward}\n"
                        f"{'='*60}"
                    )

                reward_extra_info["length_reward"] = overlong_reward * score
                reward_extra_info["response_length"] = float(n_len)
                reward_extra_info["median_correct_b"] = 0.0 if np.isnan(b) else float(b)

            final_results.append({"reward_score": reward, "reward_extra_info": reward_extra_info})

        return final_results

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item. In single mode, b falls back to config value
        since we cannot compute a meaningful median from one sample."""
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}
        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            L_max = self.overlong_buffer_cfg.get("L_max", self.max_resp_len)
            beta = self.overlong_buffer_cfg.get("beta", 1.0)
            lambda_penalty = self.overlong_buffer_cfg.get("lambda", 0.005)

            n = valid_response_length.item()
            # Single mode: no batch context, fall back to config value
            b = self.overlong_buffer_cfg.len

            overlong_reward = self._compute_length_reward(n, b, L_max, beta, lambda_penalty)
            reward += overlong_reward
            logger.warning(
                f"\n{'='*20} HMPO single DEBUG {'='*20}\n"
                f"Current length n: {n}\n"
                f"Fallback b (config): {b}\n"
                f"Max length L_max: {L_max}\n"
                f"Overlong reward: {overlong_reward}\n"
                f"Acc score: {score}\n"
                f"{'='*52}"
            )

            # Always log key metrics for tensorboard visibility
            reward_extra_info["overlong_reward"] = overlong_reward
            reward_extra_info["response_length"] = float(n)
            reward_extra_info["median_correct_b"] = float(b)

            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong"] = n >= L_max

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

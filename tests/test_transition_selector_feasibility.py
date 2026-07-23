import tempfile
import unittest
from pathlib import Path

import torch

from export_confirmatory_prefix_trajectories import build_trajectory
from utils.transition_selector import (
    FEATURE_SCHEMA, action_metrics, build_derived_trajectory, closed_loop_rollout,
    deterministic_fold_assignment, guardrail_pass, oracle_rollout, oracle_training_examples,
    pareto_frontier, paired_metrics, standardize_train_validation, train_gate,
    trajectory_metrics, transition_features,
)


def make(correct, probabilities=None, seed=3):
    correct=torch.tensor(correct,dtype=torch.bool); n,t=correct.shape
    logits=torch.full((n,t,2),-2.0); logits[...,0]=torch.where(correct,torch.tensor(2.0),torch.tensor(-2.0)); logits[...,1]=-logits[...,0]
    if probabilities is not None:
        p=torch.tensor(probabilities,dtype=torch.float32).clamp(1e-5,1-1e-5)
        logits[...,0]=torch.logit(p); logits[...,1]=0
    return build_trajectory(logits,torch.zeros(n,dtype=torch.long),method="final_ce",seed=seed,checkpoint_path="x",config={})


def good_rows(**overrides):
    rows=[]
    for seed in (3,4,5):
        row={"seed":seed,"micro_matched_regression_difference":-1.0,"candidate_matched_recovery_count":90,
             "raw_matched_recovery_count":100,"seed_recovery_preservation_ratio":.9,"final_accuracy_change":0.0,
             "mean_prefix_accuracy_change":0.0,"minimum_prefix_accuracy_change":0.0,"first_delay_mean":0.0}
        row.update(overrides); rows.append(row)
    return rows


class TransitionSelectorTests(unittest.TestCase):
    def test_01_block_oracle_keeps_c_to_w(self):
        _,log=oracle_rollout(make([[1,0]]),"oracle_block_destructive"); self.assertTrue(log["keep"][0,0])
    def test_02_block_oracle_switches_w_to_c(self):
        _,log=oracle_rollout(make([[0,1]]),"oracle_block_destructive"); self.assertFalse(log["keep"][0,0])
    def test_03_block_oracle_switches_c_to_c(self):
        _,log=oracle_rollout(make([[1,1]]),"oracle_block_destructive"); self.assertFalse(log["keep"][0,0])
    def test_04_block_oracle_switches_w_to_w(self):
        _,log=oracle_rollout(make([[0,0]]),"oracle_block_destructive"); self.assertFalse(log["keep"][0,0])
    def test_05_block_oracle_has_zero_regression(self):
        oracle,_=oracle_rollout(make([[1,0,1],[0,1,0]]),"oracle_block_destructive")
        self.assertEqual(trajectory_metrics(oracle)["correct_to_wrong_count"],0)
    def test_06_common_wrong_recovery_matches_raw(self):
        raw=make([[0,1,1],[0,0,1]]); oracle,_=oracle_rollout(raw,"oracle_block_destructive")
        metrics=paired_metrics(oracle,raw); self.assertEqual(metrics["candidate_matched_recovery_count"],metrics["raw_matched_recovery_count"])
    def test_07_first_correct_not_later(self):
        raw=make([[0,1,0],[1,0,0]]); oracle,_=oracle_rollout(raw,"oracle_block_destructive")
        from utils.prefix_metrics import first_correct_timestep
        self.assertTrue(torch.all(first_correct_timestep(oracle["prefix_logits"],oracle["targets"]) <= first_correct_timestep(raw["prefix_logits"],raw["targets"])))
    def test_08_prefix_accuracy_not_lower(self):
        raw=make([[1,0,1],[0,1,0]]); oracle,_=oracle_rollout(raw,"oracle_block_destructive")
        self.assertTrue(torch.all(oracle["correct"].sum(0)>=raw["correct"].sum(0)))
    def test_09_best_candidate_lexicographic(self):
        raw=make([[1,0],[0,1]],[[.7,.4],[.2,.6]]); _,log=oracle_rollout(raw,"oracle_best_candidate")
        self.assertEqual(log["keep"][:,0].tolist(),[True,False])
    def test_10_best_candidate_tie_switches(self):
        raw=make([[1,1]],[[.7,.7]]); _,log=oracle_rollout(raw,"oracle_best_candidate"); self.assertFalse(log["keep"][0,0])
    def test_11_derived_masks_match_logits(self):
        raw=make([[1,0]]); derived=build_derived_trajectory(raw,raw["prefix_logits"].clone(),"derived",{})
        self.assertTrue(torch.equal(derived["predictions"],derived["prefix_logits"].argmax(-1)))
        self.assertTrue(torch.equal(derived["correct"],derived["predictions"].eq(derived["targets"][:,None])))
    def test_12_features_are_target_free(self):
        self.assertTrue(all(not row["uses_target"] for row in FEATURE_SCHEMA)); self.assertNotIn("true_class_probability",[r["name"] for r in FEATURE_SCHEMA])
    def test_13_same_sample_across_seeds_same_fold(self):
        assignment=deterministic_fold_assignment(torch.tensor([0,1,2]),5,2026)
        self.assertEqual({assignment[1] for _ in (3,4,5)},{assignment[1]})
    def test_14_fold_groups_do_not_overlap(self):
        a=deterministic_fold_assignment(torch.arange(100),5,2026)
        groups=[{k for k,v in a.items() if v==fold} for fold in range(5)]
        self.assertTrue(all(groups[i].isdisjoint(groups[j]) for i in range(5) for j in range(i)))
    def test_15_normalization_uses_train_only(self):
        train=torch.tensor([[0.],[2.]]); val=torch.tensor([[100.]])
        _,scaled,mean,std=standardize_train_validation(train,val)
        self.assertEqual(mean.item(),1.); self.assertEqual(std.item(),1.); self.assertEqual(scaled.item(),99.)
    def test_16_no_positive_support_fails(self):
        with self.assertRaisesRegex(ValueError,"insufficient_support"):
            train_gate("linear_gate",torch.randn(4,13),torch.zeros(4,dtype=torch.bool),epochs=1)
    def test_17_always_switch_equals_raw(self):
        raw=make([[1,0,1],[0,1,1]]); derived,_=closed_loop_rollout(raw,lambda x:torch.zeros(x.shape[0]),.5,"always_switch")
        self.assertTrue(torch.equal(raw["prefix_logits"],derived["prefix_logits"]))
    def test_18_closed_loop_uses_accepted_state(self):
        raw=make([[1,0,1]],[[.9,.2,.8]]); seen=[]
        def fn(x): seen.append(x.clone()); return torch.ones(x.shape[0]) if len(seen)==1 else torch.zeros(x.shape[0])
        closed_loop_rollout(raw,fn,.5,"gate")
        expected,_=transition_features(raw["prefix_logits"][:,0],raw["prefix_logits"][:,2],3,3)
        self.assertTrue(torch.allclose(seen[1],expected))
    def test_19_threshold_keep_count_monotone(self):
        p=torch.tensor([.1,.5,.9]); counts=[int((p>=x).sum()) for x in (.1,.5,.95)]; self.assertEqual(counts,sorted(counts,reverse=True))
    def test_20_recovery_failure_not_promising(self):
        self.assertFalse(guardrail_pass(good_rows(seed_recovery_preservation_ratio=.7,candidate_matched_recovery_count=70),.5))
    def test_21_prefix_failure_not_promising(self):
        self.assertFalse(guardrail_pass(good_rows(mean_prefix_accuracy_change=-1.1),.5))
    def test_22_one_seed_improvement_not_promising(self):
        rows=good_rows(); rows[1]["micro_matched_regression_difference"]=1; rows[2]["micro_matched_regression_difference"]=1
        self.assertFalse(guardrail_pass(rows,.1))
    def test_23_two_of_three_and_aggregate_required(self):
        rows=good_rows(); rows[2]["micro_matched_regression_difference"]=1
        self.assertTrue(guardrail_pass(rows,.1)); self.assertFalse(guardrail_pass(rows,1.0))
    def test_24_seeded_fold_and_training_deterministic(self):
        self.assertEqual(deterministic_fold_assignment(torch.arange(20)),deterministic_fold_assignment(torch.arange(20)))
        x=torch.randn(20,13); y=torch.tensor([0,1]*10,dtype=torch.bool)
        m1,_=train_gate("linear_gate",x,y,epochs=2,batch_size=20,seed=7); m2,_=train_gate("linear_gate",x,y,epochs=2,batch_size=20,seed=7)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(m1.parameters(),m2.parameters())))
    def test_25_existing_mechanism_summary_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"mechanism_summary.json"; path.write_text("sentinel"); oracle_rollout(make([[1,0]]),"oracle_block_destructive")
            self.assertEqual(path.read_text(),"sentinel")

if __name__ == "__main__": unittest.main()

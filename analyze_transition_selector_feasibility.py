#!/usr/bin/env python3
"""Post-hoc transition-selector feasibility analysis for N-MNIST final-CE trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.transition_selector import (
    FEATURE_SCHEMA, TRANSITION_TYPES, action_metrics, aggregate_seed_metrics,
    build_derived_trajectory, closed_loop_rollout, deterministic_fold_assignment,
    guardrail_pass, intervention_metrics, oracle_rollout, oracle_training_examples,
    paired_metrics, pareto_frontier, standardize_train_validation, train_gate,
    validate_final_ce_trajectories,
)

SEEDS=(3,4,5)
ORACLES=("oracle_block_destructive","oracle_best_candidate")
GATES=("linear_gate","mlp16_gate")
THRESHOLDS=tuple(round(x/20,2) for x in range(1,20))
COLORS={"raw_final_ce":"#4C78A8","oracle_block_destructive":"#F58518","oracle_best_candidate":"#54A24B","linear_gate":"#E45756","mlp16_gate":"#72B7B2","confidence_hysteresis":"#B279A2","always_switch":"#9D755D"}


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True)+"\n")


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:("" if isinstance(v,float) and not math.isfinite(v) else v) for k,v in row.items()})


def load_inputs(root: Path) -> tuple[list[dict[str,Any]],dict[str,Any],list[str]]:
    paths=[root/"mechanism_analysis"/"trajectories"/"final_ce"/f"seed_{seed}.pt" for seed in SEEDS]
    missing=[str(path) for path in paths if not path.is_file()]
    mechanism=root/"mechanism_analysis"/"mechanism_summary.json"
    if not mechanism.is_file(): missing.append(str(mechanism))
    if missing: raise FileNotFoundError("Missing required input(s); run trajectory export/mechanism analysis first:\n"+"\n".join(missing))
    trajectories=[torch.load(path,map_location="cpu",weights_only=False) for path in paths]
    validate_final_ce_trajectories(trajectories)
    return trajectories,json.loads(mechanism.read_text()),[str(path) for path in paths]+[str(mechanism)]


def macro_regression_difference(candidate: dict[str,Any], raw: dict[str,Any]) -> float:
    values=[]
    cc=candidate["correct"].bool(); rc=raw["correct"].bool()
    for t in range(cc.shape[1]-1):
        support=cc[:,t]&rc[:,t]
        if support.any():
            values.append(100.0*float(((support&~cc[:,t+1]).sum()-(support&~rc[:,t+1]).sum()))/int(support.sum()))
    return sum(values)/len(values) if values else float("nan")


def evaluated_row(candidate,raw,method,seed,actions=None,codes=None,**extra):
    row={"method":method,"seed":seed,**extra,**paired_metrics(candidate,raw)}
    row["macro_matched_regression_difference"]=macro_regression_difference(candidate,raw)
    if actions is not None:
        row.update(intervention_metrics(actions,codes))
        for t in range(actions.shape[1]): row[f"keep_rate_t{t+1}_to_t{t+2}"]=float(actions[:,t].float().mean())
        row["C_TO_W_keep_recall"]=row["keep_rate_C_TO_W"]
        row["W_TO_C_false_keep_rate"]=row["keep_rate_W_TO_C"]
    return row


def transition_rows(trajectory,method,seed):
    c=trajectory["correct"].bool(); rows=[]
    for t in range(c.shape[1]-1):
        prev=c[:,t]; new=c[:,t+1]
        rows.append({"method":method,"seed":seed,"from_t":t+1,"to_t":t+2,
          "correct_to_wrong_count":int((prev&~new).sum()),"wrong_to_correct_count":int((~prev&new).sum()),
          "previous_correct_support":int(prev.sum()),"previous_wrong_support":int((~prev).sum()),
          "conditional_regression_rate":100*float((prev&~new).sum())/int(prev.sum()) if prev.any() else float("nan"),
          "conditional_recovery_rate":100*float((~prev&new).sum())/int((~prev).sum()) if (~prev).any() else float("nan")})
    return rows


def assert_block_invariants(raw,derived):
    rc=raw["correct"].bool(); dc=derived["correct"].bool()
    failures=[]
    if int((dc[:,:-1]&~dc[:,1:]).sum()): failures.append("nonzero derived C_TO_W")
    if not torch.all(dc.sum(0)>=rc.sum(0)): failures.append("prefix correct count decreased")
    if int(dc[:,-1].sum())<int(rc[:,-1].sum()): failures.append("final correct count decreased")
    first=lambda c: torch.where(c.any(1),c.float().argmax(1)+1,torch.full((c.shape[0],),c.shape[1]+1))
    if not torch.all(first(dc)<=first(rc)): failures.append("first-correct became later")
    common_wrong=(~dc[:,:-1])&(~rc[:,:-1])
    if int((common_wrong&dc[:,1:]).sum()) != int((common_wrong&rc[:,1:]).sum()): failures.append("matched recovery changed")
    metric=paired_metrics(derived,raw)
    if metric["correct_to_wrong_count"] or metric["ever_regressed_fraction"] or metric["population_destructive_rate"] or metric["conditional_regression_rate"]:
        failures.append("destructive metric is nonzero")
    if failures: raise AssertionError("oracle_block_destructive invariant failure: "+"; ".join(failures))


def run_oracles(trajectories,out):
    seed_rows=[]; per_transition=[]; action_rows=[]; derived_by_method={"raw_final_ce":trajectories}
    for raw in trajectories:
        seed=int(raw["seed"])
        seed_rows.append(evaluated_row(raw,raw,"raw_final_ce",seed,torch.zeros_like(raw["correct"][:,:-1]),torch.zeros_like(raw["correct"][:,:-1],dtype=torch.long)))
        per_transition.extend(transition_rows(raw,"raw_final_ce",seed))
    for oracle in ORACLES:
        derived_by_method[oracle]=[]
        for raw in trajectories:
            seed=int(raw["seed"]); derived,log=oracle_rollout(raw,oracle)
            if oracle=="oracle_block_destructive": assert_block_invariants(raw,derived)
            derived_by_method[oracle].append(derived)
            seed_rows.append(evaluated_row(derived,raw,oracle,seed,log["keep"],log["transition_type_code"]))
            per_transition.extend(transition_rows(derived,oracle,seed))
            for t in range(log["keep"].shape[1]):
                for code,name in enumerate(("W_TO_W","W_TO_C","C_TO_W","C_TO_C")):
                    mask=log["transition_type_code"][:,t]==code
                    action_rows.append({"method":oracle,"seed":seed,"from_t":t+1,"to_t":t+2,"transition_type":name,
                      "support":int(mask.sum()),"keep_count":int((log["keep"][:,t]&mask).sum()),
                      "keep_rate":float(log["keep"][:,t][mask].float().mean()) if mask.any() else float("nan"),
                      "switch_rate":float((~log["keep"][:,t][mask]).float().mean()) if mask.any() else float("nan")})
            cache=out/"cache"/oracle; cache.mkdir(parents=True,exist_ok=True)
            type_names=("W_TO_W","W_TO_C","C_TO_W","C_TO_C")
            torch.save({**log,
              "transition_type":[[type_names[int(code)] for code in row] for row in log["transition_type_code"]],
              "action":[["KEEP" if bool(value) else "SWITCH" for value in row] for row in log["keep"]],
              "transition_type_mapping":{0:"W_TO_W",1:"W_TO_C",2:"C_TO_W",3:"C_TO_C"},"action_mapping":{False:"SWITCH",True:"KEEP"}},cache/f"seed_{seed}_actions.pt")
    return seed_rows,per_transition,action_rows,derived_by_method


def combine_fold_rollouts(raw,fold_models,fold_for_sample,threshold,method):
    logits=torch.empty_like(raw["prefix_logits"]); keep=torch.empty_like(raw["correct"][:,:-1]); codes=torch.empty_like(keep,dtype=torch.long); probs=torch.empty_like(keep,dtype=torch.float)
    for fold,model,mean,std in fold_models:
        mask=torch.tensor([fold_for_sample[int(i)]==fold for i in raw["sample_index"]],dtype=torch.bool)
        fn=lambda x,m=model,mu=mean,sd=std: torch.sigmoid(m((x-mu)/sd))
        derived,log=closed_loop_rollout(raw,fn,threshold,method,mask)
        logits[mask]=derived["prefix_logits"]; keep[mask]=log["keep"]; codes[mask]=log["transition_type_code"]; probs[mask]=log["probability"]
    return build_derived_trajectory(raw,logits,method,{"threshold":threshold,"evaluation":"sample-index-grouped out-of-fold closed loop"}),{"keep":keep,"transition_type_code":codes,"probability":probs}


def run_gate_cv(trajectories,out,fold_count,cv_seed,epochs,batch_size,device,required):
    examples=[oracle_training_examples(raw) for raw in trajectories]
    assignment=deterministic_fold_assignment(trajectories[0]["sample_index"],fold_count,cv_seed)
    x=torch.cat([e["features"].reshape(-1,len(FEATURE_SCHEMA)) for e in examples]); y=torch.cat([e["target"].reshape(-1) for e in examples]); codes=torch.cat([e["transition_type_code"].reshape(-1) for e in examples])
    groups=torch.cat([e["sample_index"][:,None].expand(-1,7).reshape(-1) for e in examples]); folds=torch.tensor([assignment[int(i)] for i in groups])
    training=[]; normalization=[]; teacher=[]; closed=[]; probability_summary=[]
    for model_name in GATES:
        oof=torch.empty(y.shape,dtype=torch.float); fold_models=[]
        for fold in range(fold_count):
            train=folds!=fold; valid=~train
            xtrain,xvalid,mean,std=standardize_train_validation(x[train],x[valid])
            model,info=train_gate(model_name,xtrain,y[train],epochs=epochs,batch_size=batch_size,seed=cv_seed+fold,device=device)
            with torch.no_grad(): oof[valid]=torch.sigmoid(model(xvalid))
            fold_models.append((fold,model,mean,std))
            training.append({"model":model_name,"fold":fold,**info})
            for feature,mu,sd in zip(FEATURE_SCHEMA,mean.tolist(),std.tolist()):
                normalization.append({"model":model_name,"fold":fold,"feature":feature["name"],"train_mean":mu,"train_std":sd})
            teacher.append({"model":model_name,"scope":"fold","fold":fold,**action_metrics(oof[valid],y[valid],codes[valid])})
        teacher.append({"model":model_name,"scope":"all_oof","fold":"all",**action_metrics(oof,y,codes)})
        for code,name in enumerate(("W_TO_W","W_TO_C","C_TO_W","C_TO_C")):
            mask=codes==code
            probability_summary.append({"model":model_name,"transition_type":name,"support":int(mask.sum()),"mean_probability":float(oof[mask].mean()),"std_probability":float(oof[mask].std(unbiased=False))})
        for threshold in THRESHOLDS:
            seed_rows=[]
            for raw in trajectories:
                derived,log=combine_fold_rollouts(raw,fold_models,assignment,threshold,model_name)
                seed_rows.append(evaluated_row(derived,raw,model_name,int(raw["seed"]),log["keep"],log["transition_type_code"],threshold=threshold,row_scope="seed"))
            passed=guardrail_pass(seed_rows,required); agg=aggregate_seed_metrics(seed_rows)
            closed.extend(seed_rows); closed.append({"method":model_name,"seed":"aggregate","threshold":threshold,"row_scope":"aggregate","guardrail_pass":passed,**agg})
    return training,normalization,teacher,closed,probability_summary,assignment


def run_heuristics(trajectories,required):
    rows=[]
    configs=[("always_switch",None,None)]+[("confidence_hysteresis",p,n) for p in (.70,.80,.90,.95) for n in (.50,.60,.70,.80)]
    for name,p,n in configs:
        seed_rows=[]
        for raw in trajectories:
            if name=="always_switch": fn=lambda x: torch.zeros(x.shape[0]); threshold=.5
            else: fn=lambda x,pp=p,nn=n: ((x[:,6]<.5)&(x[:,0]>=pp)&(x[:,3]<=nn)).float(); threshold=.5
            derived,log=closed_loop_rollout(raw,fn,threshold,name)
            if name=="always_switch" and not torch.equal(derived["prefix_logits"],raw["prefix_logits"]): raise AssertionError("always_switch must exactly reproduce raw_final_ce")
            seed_rows.append(evaluated_row(derived,raw,name,int(raw["seed"]),log["keep"],log["transition_type_code"],previous_confidence_threshold=p,new_confidence_threshold=n,row_scope="seed"))
        passed=guardrail_pass(seed_rows,required); agg=aggregate_seed_metrics(seed_rows)
        rows.extend(seed_rows); rows.append({"method":name,"seed":"aggregate","row_scope":"aggregate","previous_confidence_threshold":p,"new_confidence_threshold":n,"guardrail_pass":passed,**agg})
    return rows


def plot_outputs(out,derived,action_rows,probability_rows,closed,heuristics,pareto):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,ax=plt.subplots(figsize=(8,5))
    for method,runs in derived.items():
        curves=torch.tensor([r["correct"].float().mean(0).tolist() for r in runs]).mean(0)*100
        ax.plot(range(1,9),curves,label=method,color=COLORS[method],marker="o")
    ax.set(title="Oracle upper-bound prefix accuracy",xlabel="Prefix timestep",ylabel="Accuracy (%)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"oracle_prefix_accuracy_curves.png",dpi=180); plt.close(fig)
    aggregate=[]
    for oracle in ORACLES:
        for typ in TRANSITION_TYPES:
            vals=[r["keep_rate"] for r in action_rows if r["method"]==oracle and r["transition_type"]==typ and math.isfinite(r["keep_rate"])]
            aggregate.append((oracle,typ,sum(vals)/len(vals)))
    fig,ax=plt.subplots(figsize=(8,5)); x=torch.arange(4).numpy(); width=.35
    for i,oracle in enumerate(ORACLES): ax.bar(x+(i-.5)*width,[v for m,t,v in aggregate if m==oracle],width,label=oracle,color=COLORS[oracle])
    ax.set_xticks(x,TRANSITION_TYPES); ax.set_ylim(0,1); ax.set(title="Oracle action rates by transition type",ylabel="KEEP rate"); ax.legend(); fig.tight_layout(); fig.savefig(out/"oracle_transition_action_rates.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    if probability_rows:
        for i,model in enumerate(GATES): ax.bar(x+(i-.5)*width,[next(r["mean_probability"] for r in probability_rows if r["model"]==model and r["transition_type"]==t) for t in TRANSITION_TYPES],width,label=model,color=COLORS[model])
        ax.legend()
    else:
        ax.text(.5,.5,"Gate CV skipped",ha="center",va="center",transform=ax.transAxes)
    ax.set_xticks(x,TRANSITION_TYPES); ax.set_ylim(0,1); ax.set(title="Out-of-fold teacher-forced gate scores",ylabel="Mean P(KEEP)"); fig.tight_layout(); fig.savefig(out/"gate_action_probability_by_transition_type.png",dpi=180); plt.close(fig)
    configs=[r for r in closed+heuristics if r.get("row_scope")=="aggregate"]
    fig,ax=plt.subplots(figsize=(8,5)); seen=set()
    for r in configs:
        m=r["method"]; label=m if m not in seen else None; seen.add(m)
        ax.scatter(r.get("pooled_matched_recovery_preservation",float("nan")),-r.get("micro_matched_regression_difference_mean",float("nan")),color=COLORS[m],marker="*" if r.get("guardrail_pass") else "o",alpha=.75,label=label)
    ax.axvline(.9,color="gray",ls="--"); ax.axhline(0,color="black",lw=.8); ax.set(title="Closed-loop regression–recovery trade-off",xlabel="Pooled matched recovery preservation",ylabel="Regression improvement (percentage points)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"gate_regression_recovery_tradeoff.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); seen=set()
    for r in configs:
        m=r["method"]; label=m if m not in seen else None; seen.add(m)
        ax.scatter(r.get("mean_prefix_accuracy_change_mean",float("nan")),r.get("final_accuracy_change_mean",float("nan")),color=COLORS[m],marker="*" if r.get("guardrail_pass") else "o",alpha=.75,label=label)
    ax.axvline(-1,color="gray",ls="--"); ax.axhline(-.1,color="gray",ls="--"); ax.set(title="Closed-loop prefix guardrails",xlabel="Mean prefix accuracy change (pp)",ylabel="Final accuracy change (pp)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"gate_prefix_guardrail_tradeoff.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    groups={"learned":[r for r in configs if r["method"] in GATES],"heuristic":[r for r in configs if r["method"] not in GATES]}
    for label,rs in groups.items(): ax.scatter([r["pooled_matched_recovery_preservation"] for r in rs],[-r["micro_matched_regression_difference_mean"] for r in rs],label=label,alpha=.75)
    ax.axhline(0,color="black",lw=.8); ax.set(title="Heuristic versus learned selectors",xlabel="Pooled matched recovery preservation",ylabel="Regression improvement (pp)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"heuristic_vs_learned_gate.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-root",type=Path,default=Path("results/temporal_reliability_nmnist_confirmatory"))
    parser.add_argument("--cv-folds",type=int,default=5); parser.add_argument("--cv-seed",type=int,default=2026)
    parser.add_argument("--device",default="cpu"); parser.add_argument("--epochs",type=int,default=300); parser.add_argument("--batch-size",type=int,default=4096)
    parser.add_argument("--skip-gate-cv",action="store_true")
    args=parser.parse_args(); out=args.results_root/"transition_selector_analysis"; out.mkdir(parents=True,exist_ok=True)
    trajectories,mechanism,input_paths=load_inputs(args.results_root)
    required=0.5*abs(float(mechanism["matched_regression_difference_mean"]))
    oracle_rows,transition_rows_,action_rows,derived=run_oracles(trajectories,out)
    write_csv(out/"oracle_per_seed_metrics.csv",oracle_rows); write_csv(out/"oracle_per_transition_metrics.csv",transition_rows_); write_csv(out/"oracle_action_summary.csv",action_rows)
    write_json(out/"feature_schema.json",{"features":FEATURE_SCHEMA,"target":"oracle_block_destructive KEEP action","target_free_at_inference":True})
    training=[]; normalization=[]; teacher=[]; closed=[]; probability=[]; assignment={}
    if not args.skip_gate_cv:
        training,normalization,teacher,closed,probability,assignment=run_gate_cv(trajectories,out,args.cv_folds,args.cv_seed,args.epochs,args.batch_size,args.device,required)
    write_csv(out/"gate_fold_training_summary.csv",training); write_csv(out/"feature_normalization_by_fold.csv",normalization)
    write_csv(out/"gate_teacher_forced_metrics.csv",teacher); write_csv(out/"gate_closed_loop_threshold_sweep.csv",closed); write_json(out/"cv_fold_assignment.json",{"cv_seed":args.cv_seed,"fold_count":args.cv_folds,"sample_index_to_fold":assignment})
    heuristic=run_heuristics(trajectories,required); write_csv(out/"heuristic_threshold_sweep.csv",heuristic)
    configs=[r for r in closed+heuristic if r.get("row_scope")=="aggregate"]
    pareto_input=[{"method":r["method"],"threshold":r.get("threshold"),"previous_confidence_threshold":r.get("previous_confidence_threshold"),"new_confidence_threshold":r.get("new_confidence_threshold"),"regression_improvement":-r.get("micro_matched_regression_difference_mean",float("nan")),"pooled_matched_recovery_preservation":r.get("pooled_matched_recovery_preservation",float("nan")),"mean_prefix_accuracy_change":r.get("mean_prefix_accuracy_change_mean",float("nan")),"final_accuracy_change":r.get("final_accuracy_change_mean",float("nan")),"guardrail_pass":r.get("guardrail_pass",False)} for r in configs]
    finite=[r for r in pareto_input if all(math.isfinite(float(r[k])) for k in ("regression_improvement","pooled_matched_recovery_preservation","mean_prefix_accuracy_change"))]
    frontier=pareto_frontier(finite); frontier_ids={id(r) for r in frontier}
    for r in finite: r["pareto_optimal"]=id(r) in frontier_ids
    write_csv(out/"pareto_frontier.csv",finite)
    block=[r for r in oracle_rows if r["method"]=="oracle_block_destructive"]
    block_agg=aggregate_seed_metrics(block)
    oracle_feasible=(
      all(r["correct_to_wrong_count"]==0 for r in block)
      and all(r["candidate_matched_recovery_count"]==r["raw_matched_recovery_count"] for r in block)
      and all(r["seed_recovery_preservation_ratio"]==1.0 for r in block)
      and all(r["final_accuracy_change"]>=0 and r["mean_prefix_accuracy_change"]>=0 and r["minimum_prefix_accuracy_change"]>=0 and r["first_delay_mean"]<=0 for r in block)
      and block_agg["pooled_matched_recovery_preservation"]==1.0)
    all_passing=[r for r in configs if r.get("guardrail_pass")]
    gate_passing=[r for r in all_passing if r["method"] in GATES]
    heuristic_passing=[r for r in all_passing if r["method"] not in GATES]
    best=max(configs,key=lambda r:-r.get("micro_matched_regression_difference_mean",float("inf"))) if configs else None
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    summary={
      "analysis_type":"post_hoc_transition_selector_feasibility","evidence_scope":"Same N-MNIST test trajectories; exploratory upper-bound and cross-validated post-hoc evidence, not independent confirmation.",
      "source_commit":commit,"source_results_root":str(args.results_root),"source_trajectory_paths":input_paths[:3],"input_files":input_paths,"input_validation":{"status":"passed","seeds":list(SEEDS),"shape":list(trajectories[0]["prefix_logits"].shape),"targets_and_sample_order_aligned":True},
      "oracle_analysis_type":"label-informed upper-bound feasibility analysis","oracle_feasible":oracle_feasible,"oracle_invariant_checks":{"status":"passed","checks":["zero accepted C_TO_W","exact common-wrong recovery preservation","first-correct no later","prefix accuracy no lower","final accuracy no lower"]},
      "oracle_metrics_by_seed":{oracle:[r for r in oracle_rows if r["method"]==oracle] for oracle in ORACLES},"oracle_aggregate_metrics":{oracle:aggregate_seed_metrics([r for r in oracle_rows if r["method"]==oracle]) for oracle in ORACLES},
      "raw_final_ce_reference":{"matched_regression_difference_mean_from_mechanism":mechanism["matched_regression_difference_mean"],"required_absolute_improvement_pp":required},
      "oracle_block_destructive":{"feasible":oracle_feasible,"aggregate":block_agg,"invariants":"passed"},
      "oracle_best_candidate":{"aggregate":aggregate_seed_metrics([r for r in oracle_rows if r["method"]=="oracle_best_candidate"])},
      "feature_schema_file":"feature_schema.json","feature_names":[row["name"] for row in FEATURE_SCHEMA],"target_free_feature_check":all(not row["uses_target"] for row in FEATURE_SCHEMA),
      "cv_type":"5-fold grouped cross-validation","cv_seed":args.cv_seed,"cv_fold_count":args.cv_folds,"grouping_unit":"sample_index","cv_protocol":{"folds":args.cv_folds,"group":"sample_index","seed":args.cv_seed,"teacher_forced_training":True,"closed_loop_evaluation":True,"epochs":args.epochs,"batch_size":args.batch_size,"device":args.device,"skipped":args.skip_gate_cv},
      "gate_models":list(GATES),"thresholds":list(THRESHOLDS),"heuristic_grid":{"previous_conf_threshold":[.70,.80,.90,.95],"new_conf_threshold":[.50,.60,.70,.80]},"teacher_forced_metrics":[r for r in teacher if r.get("scope")=="all_oof"],"closed_loop_default_threshold_metrics":[r for r in closed if r.get("row_scope")=="aggregate" and r.get("threshold")==.5],"guardrails":{"regression_improved_seeds_min":2,"mean_regression_improvement_pp_min":required,"pooled_recovery_min":.9,"per_seed_recovery_min":.8,"final_accuracy_change_pp_min":-.1,"mean_prefix_accuracy_change_pp_min":-1,"minimum_prefix_accuracy_change_pp_min":-3,"mean_first_correct_delay_max":.1},
      "passing_configuration_count":len(gate_passing),"exploratory_feasible_threshold_count":len(gate_passing),"exploratory_feasible_threshold_count_by_model":{model:sum(r.get("guardrail_pass",False) for r in configs if r["method"]==model) for model in GATES},"diagnostic_status_by_model":{model:("gate_diagnostic_promising" if any(r.get("guardrail_pass",False) for r in configs if r["method"]==model) else "gate_diagnostic_not_promising") for model in GATES},"passing_configurations":gate_passing,"heuristic_guardrail_pass_count":len(heuristic_passing),"heuristic_passing_configurations":heuristic_passing,"all_guardrail_passing_configurations":all_passing,"best_regression_configuration":best,"pareto_configuration_count":len(frontier),"pareto_configurations":frontier,
      "conclusion":{"oracle_feasibility":"feasible" if oracle_feasible else "not_feasible","gate_diagnostic":"not_evaluated" if args.skip_gate_cv else ("gate_diagnostic_promising" if gate_passing else "gate_diagnostic_not_promising"),"heuristic_diagnostic":"exploratory_guardrail_pass" if heuristic_passing else "no_guardrail_pass","practical_selector_feasibility":"heuristic_only_exploratory_support" if heuristic_passing and not gate_passing else ("learned_gate_exploratory_support" if gate_passing else ("not_evaluated" if args.skip_gate_cv else "not_supported_under_prespecified_guardrails"))},
      "limitations":["The oracle uses ground-truth labels and is not deployable.","The gate diagnostic reuses the previously observed N-MNIST test trajectories.","Grouped cross-validation prevents sample leakage but does not create independent test evidence.","No threshold or model from this analysis should be treated as a finalized deployment choice.","A future method-development experiment must train and calibrate the gate using training/validation data and evaluate once on an untouched test set."],
      "artifacts":["oracle_per_seed_metrics.csv","oracle_per_transition_metrics.csv","oracle_action_summary.csv","gate_fold_training_summary.csv","gate_teacher_forced_metrics.csv","gate_closed_loop_threshold_sweep.csv","heuristic_threshold_sweep.csv","pareto_frontier.csv","feature_normalization_by_fold.csv","cv_fold_assignment.json","feature_schema.json"]}
    write_json(out/"transition_selector_summary.json",summary)
    plot_outputs(out,derived,action_rows,probability,closed,heuristic,finite)
    print(json.dumps(safe(summary["conclusion"]),indent=2)); print(f"Wrote analysis to {out}")

if __name__=="__main__": main()

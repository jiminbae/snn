#!/usr/bin/env python3
"""Aggregate fixed N-MNIST confirmatory temporal-reliability runs."""
import argparse,csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REQUIRED=("final_ce","symmetric_kl","selective_regression_thr0.6")
METRICS=("final_accuracy","mean_prefix_accuracy","minimum_prefix_accuracy","ever_regressed_fraction","mean_population_regression","mean_conditional_regression","correct_to_wrong_transition_count","destructive_transition_fraction","ever_recovered_fraction","wrong_to_correct_transition_count","beneficial_transition_fraction","mean_stable_correct_timestep","stable_by_t4_fraction","stable_by_t6_fraction","never_stable_fraction")
def stats(xs):
 m=sum(xs)/len(xs); return {"mean":m,"sample_standard_deviation":math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else 0.,"individual_seed_values":xs,"valid_seed_count":len(xs)}
def name(s):
 return f"selective_regression_thr{float(s.get('temporal_confidence_threshold',float('nan'))):g}" if s["temporal_training_mode"]=="selective_regression" else s["temporal_training_mode"]
def matched(rows):
 d={(r["method"],r["seed"]):r for r in rows}; seeds=sorted({r["seed"] for r in rows})
 return [s for s in seeds if all((m,s) in d for m in REQUIRED)],d
def recommendation_from_records(rows):
 seeds,d=matched(rows)
 if len(seeds)<3:return "no_go",[f"Matched seed count: {len(seeds)} (go requires 3).","No statistical significance is claimed."]
 f=[d[("final_ce",s)] for s in seeds]; q=[d[("selective_regression_thr0.6",s)] for s in seeds]; k=[d[("symmetric_kl",s)] for s in seeds]
 mean=lambda rs,key:stats([r[key] for r in rs])["mean"]; n=sum(x["ever_regressed_fraction"]<y["ever_regressed_fraction"] for x,y in zip(q,f))
 rr=mean(q,"ever_regressed_fraction")<mean(f,"ever_regressed_fraction"); br=mean(q,"beneficial_transition_fraction")/max(1e-8,mean(f,"beneficial_transition_fraction")); er=mean(q,"ever_recovered_fraction")/max(1e-8,mean(f,"ever_recovered_fraction")); ac=mean(q,"final_accuracy")-mean(f,"final_accuracy")
 sk=mean(q,"destructive_transition_fraction")<mean(k,"destructive_transition_fraction") and mean(q,"beneficial_transition_fraction")>mean(k,"beneficial_transition_fraction")
 strict=n>=2 and rr and br>=.9 and er>=.9 and ac>=-.5 and sk; hard=n<2 or not rr or br<.8 or er<.8 or ac<-1
 return ("go" if strict else "weak_go" if not hard else "no_go"),[f"Matched seeds: {seeds}; regression reduced in {n} seed(s).",f"Regression={rr}; beneficial preservation={br:.4f}; recovered preservation={er:.4f}; final accuracy change={ac:.4f} pp.",f"Selective beats symmetric KL on both transition directions: {sk}.","No statistical significance is claimed."]
def write(path,rows):
 fields=sorted({k for r in rows for k in r}) if rows else []
 with path.open("w",newline="",encoding="utf8") as h:
  w=csv.DictWriter(h,fieldnames=fields)
  if fields:w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dirs",nargs="+",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); rows=[]
 for rd in map(Path,a.run_dirs):
  s=json.loads((rd/"temporal_reliability_summary.json").read_text()); c=json.loads((rd/"config.json").read_text()); rows.append({"method":name(s),"seed":int(c["seed"]),"prefix_accuracy_curve":s["prefix_accuracy_curve"],**{m:float(s[m]) for m in METRICS if m in s}})
 seeds,d=matched(rows); active=[d[(m,s)] for m in REQUIRED for s in seeds]; metrics=[m for m in METRICS if all(m in r for r in active)]
 mr=[{"method":m,"metric":x,**stats([d[(m,s)][x] for s in seeds])} for m in REQUIRED for x in metrics];write(out/"aggregate_temporal_metrics.csv",mr)
 cmp=[]
 for s in seeds:
  f,k,q=(d[(m,s)] for m in REQUIRED)
  for m,r in (("symmetric_kl",k),("selective_regression_thr0.6",q)):
   z={"method":m,"seed":s,"regression_reduction_vs_final_ce":f["ever_regressed_fraction"]-r["ever_regressed_fraction"],"final_accuracy_change_vs_final_ce":r["final_accuracy"]-f["final_accuracy"],"mean_prefix_accuracy_change_vs_final_ce":r["mean_prefix_accuracy"]-f["mean_prefix_accuracy"],"minimum_prefix_accuracy_change_vs_final_ce":r["minimum_prefix_accuracy"]-f["minimum_prefix_accuracy"],"beneficial_transition_preservation_ratio":r["beneficial_transition_fraction"]/max(1e-8,f["beneficial_transition_fraction"]),"ever_recovered_preservation_ratio":r["ever_recovered_fraction"]/max(1e-8,f["ever_recovered_fraction"])}
   if m.startswith("selective"):z.update({"destructive_transition_change_vs_symmetric_kl":r["destructive_transition_fraction"]-k["destructive_transition_fraction"],"beneficial_transition_change_vs_symmetric_kl":r["beneficial_transition_fraction"]-k["beneficial_transition_fraction"]})
   cmp.append(z)
 write(out/"seed_method_comparisons.csv",cmp); write(out/"aggregate_method_comparisons.csv",[{"method":m,"metric":x,**stats([r[x] for r in cmp if r["method"]==m and x in r])} for m in ("symmetric_kl","selective_regression_thr0.6") for x in sorted({x for r in cmp if r["method"]==m for x in r}-{"method","seed"})])
 dec,reasons=recommendation_from_records(rows);(out/"aggregate_summary.json").write_text(json.dumps({"recommendation":dec,"recommendation_reasons":reasons,"required_methods":REQUIRED,"matched_seeds":seeds,"matched_seed_count":len(seeds)},indent=2))
 for file,x,y in (("prefix_accuracy_comparison.png",None,None),("regression_recovery_tradeoff.png","destructive_transition_fraction","beneficial_transition_fraction")):
  plt.figure(figsize=(7,5))
  for m in REQUIRED:
   rs=[d[(m,s)] for s in seeds]
   if x:plt.scatter([r[x] for r in rs],[r[y] for r in rs],label=m)
   elif rs:
    c=[sum(v)/len(v) for v in zip(*[r["prefix_accuracy_curve"] for r in rs])];plt.plot(range(1,len(c)+1),c,"o-",label=m)
  plt.xlabel("Destructive transitions (%)" if x else "Timestep");plt.ylabel("Beneficial transitions (%)" if x else "Accuracy (%)");plt.grid(alpha=.3);plt.legend();plt.tight_layout();plt.savefig(out/file,dpi=160);plt.close()
if __name__=="__main__":main()


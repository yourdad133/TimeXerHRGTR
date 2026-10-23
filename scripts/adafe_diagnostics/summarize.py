"""Collect prespecified diagnostic results without choosing variants by test error."""
import argparse
import csv
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    output=Path(parser.parse_args().output)
    plan=json.loads((output/"plan.json").read_text())
    rows=[]
    failures=[]
    for job in plan["jobs"]:
        case=f'{job["dataset"]}_seed{job["seed"]}_{job["variant"]}'
        directory=output/"training/runs"/case
        if not (directory/"result.json").exists():
            status=json.loads((directory/"status.json").read_text()) if (directory/"status.json").exists() else {"state":"missing"}
            failures.append({"case":case,**status})
            continue
        result=json.loads((directory/"result.json").read_text())
        diagnostic=result["best_diagnostics"]
        rows.append({"dataset":result["dataset"],"seed":result["seed"],"variant":result["variant"],
            "validation_mse":result["validation_mse"],"test_mse":result["test"]["mse"],
            "test_mae":result["test"]["mae"],"best_epoch":result["best_epoch"],
            "gamma":diagnostic.get("gamma"),"score_std":diagnostic.get("score_std"),
            "saturation":diagnostic.get("proxy_saturated_fraction"),
            "residual_ratio":diagnostic.get("residual_to_patch_norm"),
            "full_data":result["full_dataset_training"],"initial_shared_hash":result["initial_shared_hash"]})
    for row in rows:
        base=next((r for r in rows if r["dataset"]==row["dataset"] and r["seed"]==row["seed"] and r["variant"]=="baseline"),None)
        original=next((r for r in rows if r["dataset"]==row["dataset"] and r["seed"]==row["seed"] and r["variant"]=="ste"),None)
        if base and base["initial_shared_hash"]!=row["initial_shared_hash"]:
            raise AssertionError("Shared initialization differs")
        row["test_change_vs_baseline_pct"]=100*(row["test_mse"]/base["test_mse"]-1) if base else None
        row["validation_change_vs_ste_pct"]=100*(row["validation_mse"]/original["validation_mse"]-1) if original else None
        row["test_change_vs_ste_pct"]=100*(row["test_mse"]/original["test_mse"]-1) if original else None
    (output/"summary.json").write_text(json.dumps({"runs":rows,"failures":failures,"planned":len(plan["jobs"]),
         "test_used_for_selection":False},indent=2)+"\n")
    if rows:
        with (output/"summary.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"completed":len(rows),"planned":len(plan["jobs"]),"failures":len(failures)}))

if __name__=="__main__":
    main()

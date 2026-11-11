"""One request per invocation; no scientific overrides or test-image operations."""
import argparse
import json
from pathlib import Path
import signal

from .config import ROOT,contained,load_contract,read_json,policy
from .runtime import ExecutionContext,inspect_inputs,preflight,configure_runtime


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m src.e3",description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    check=commands.add_parser("check-config"); check.add_argument("--input",type=Path,required=True)
    for name in ("accept-source","extract-cache","run-head","report-selected"):
        command=commands.add_parser(name)
        command.add_argument("--request",type=Path,required=True)
        command.add_argument("--inventory",type=Path,required=True)
        command.add_argument("--output",type=Path,required=True)
        if name=="accept-source": command.add_argument("--gpu",action="store_true")
        if name=="run-head": command.add_argument("--resume",type=Path)
    study=commands.add_parser("report-study"); study.add_argument("--input",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="check-config":
            value=read_json(args.input)
            if value.get("kind")=="e3_numeric_contract_v1": value=load_contract(args.input,stage="design")
            result=inspect_inputs(value)
        elif args.command=="report-study":
            from .report import aggregate_study
            request=read_json(args.input)
            if set(request)!={"head_run_dirs","source_manifest","cache_indices","protocol"}:
                raise ValueError("Complete explicit study-report input package required")
            result=aggregate_study([Path(p) for p in request["head_run_dirs"]],read_json(contained(ROOT,request["source_manifest"])),
                                   request["cache_indices"],request["protocol"])
        else:
            context=ExecutionContext(read_json(args.request),args.inventory)
            if args.command=="accept-source":
                from .acceptance import run_acceptance
                result=run_acceptance(context,output=args.output,gpu=args.gpu)
            elif args.command=="extract-cache":
                from .sources import load_frozen_source
                from .data import FixedManifest,make_source_loader
                from .cache import extract_cache
                checked=preflight(context,operation="bulk_cache"); declaration=checked["declaration"]
                environment=configure_runtime(root_seed=declaration["source_seed"])
                from .runtime import verify_current_environment
                verify_current_environment(checked,environment)
                model,_=load_frozen_source(declaration["source"],contract=context,device="cuda:0")
                start=sum(read_json(p)["chunk"]["row_count"] for p in sorted((args.output/"chunks").glob("commit_*.json")))
                loader=make_source_loader(FixedManifest(),source=declaration["source"],request={"split":declaration["split"],
                    "stage":"final","root_seed":declaration["source_seed"],"data_root":checked["data_root"],"row_start":start})
                result=extract_cache(context,source_model=model,loader=loader,output=args.output)
            elif args.command=="run-head":
                from .train import run_head
                stopping=[False]
                def request_stop(*unused): stopping[0]=True
                for name in ("SIGTERM","SIGINT","SIGUSR1"):
                    if hasattr(signal,name): signal.signal(getattr(signal,name),request_stop)
                result=run_head(context,args.output,resume=args.resume,stop_requested=lambda:stopping[0])
                result={k:v for k,v in result.items() if k!="progress"}
            else:
                from .report import evaluate_selected
                result=evaluate_selected(context,args.output)
                result={"run_id":result["index"]["run_id"],"status":"SELECTED_ANALYSIS_VALIDATED"}
        print(json.dumps(result,ensure_ascii=True,allow_nan=False))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    raise SystemExit(main())

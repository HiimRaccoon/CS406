import math
import pytest


def rows():
    scores=("top1","top3","top5","macro_f1_supported","weighted_f1")
    result=[]
    for family in ("E1","E2"):
        for suffix,values in [("original",[.1,.2,.3]),("N",[.2,.5,.5]),("cRT",[.3,.4,.7])]:
            for seed,value in zip((2001,3001,4001),values):
                arm=family+"-original" if suffix=="original" else "E3-"+family+"-"+suffix
                result.append({"arm":arm,"seed":seed,"metrics":{**{s:value for s in scores},"nll":1.}})
    return result


def test_mean_sample_std_and_paired_delta_std():
    from src.e3.report import _summarize_study_rows,render_study
    report=_summarize_study_rows(rows())
    assert report["arms"]["E1-original"]["macro_f1_supported"]["mean"]==pytest.approx(.2)
    assert report["arms"]["E1-original"]["macro_f1_supported"]["std"]==pytest.approx(.1)
    primary=report["contrasts"]["PRIMARY:E1"]["macro_f1_supported"]
    assert primary["values"]==pytest.approx([.1,-.1,.2])
    assert primary["std"]==pytest.approx(math.sqrt(7/300))
    text=render_study(report)
    assert all(label in text for label in ["PRIMARY","SECONDARY","EXPLORATORY","20.0000","LOCKED"])
    with pytest.raises(ValueError): _summarize_study_rows(rows()[:-1])


def test_predictions_reject_reordering_and_csv_content_even_after_rehash(tmp_path):
    from src.e3.report import _validate_predictions,_tables,_validate_tables,_csv_bytes,_metrics_from_predictions
    records=[{"relative_path":"A/1.jpg","label_id":0},{"relative_path":"A/2.jpg","label_id":1}]
    predictions=[{**r,"prediction":0,"top5":[0,1,2,3,4],"nll":1.} for r in records]
    _validate_predictions(predictions,records)
    with pytest.raises(ValueError): _validate_predictions(list(reversed(predictions)),records)
    metrics=_metrics_from_predictions(predictions,{i:1 for i in range(124)},[str(i) for i in range(124)],nll=1.)
    history=[{"window":1,"presentations":5,"updates":2,"train":{"ce":1.,"lr_first":.1,"lr_last":.1},"validation":metrics}]
    tables=_tables(metrics,predictions,history,[str(i) for i in range(124)])
    for name,(data,fields) in tables.items(): (tmp_path/name).write_bytes(_csv_bytes(data,fields))
    _validate_tables(tmp_path,tables)
    p=tmp_path/"predictions.csv"; p.write_bytes(p.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError): _validate_tables(tmp_path,tables)


def test_original_reference_requires_family_schema_and_full_selected_evidence(tmp_path):
    from src.e3.report import load_source_reference
    from src.e3.io import atomic_json
    from tests.e3.helpers import declaration
    p=tmp_path/"analysis.json"
    atomic_json(p,{"kind":"e2_selected_validation_analysis_v1"})
    with pytest.raises(ValueError): load_source_reference(p,source=declaration()["source"])

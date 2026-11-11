import copy
import pytest


def trials():
    return [{"candidate_id":candidate,"family":family,"arm":arm,"source_seed":2001,"root_seed":1001,"stage":"pilot",
             "selected_metric":.2 if candidate=="adamw_lr1e3" else .2000005,"attempts":1,
             "history":[{"window":i,"train":{"ce":1.},"validation":{"macro_f1_supported":.2 if candidate=="adamw_lr1e3" else .2000005}} for i in range(1,21)]}
            for candidate in ("adamw_lr1e3","adamw_lr3e3") for family in ("E1","E2") for arm in ("natural","crt")]


def test_common_profile_global_tie_and_all_arms_adequacy():
    from src.e3.campaign import _choose_protocol
    from src.e3.config import policy
    result=_choose_protocol(trials(),policy()["pilot"])
    assert result["winner_candidate_id"]=="adamw_lr1e3" and result["status"]=="REVIEW_REQUIRED"
    assert result["official_final_authorized"] is False
    with pytest.raises(ValueError): _choose_protocol(trials()[:-1],policy()["pilot"])
    unstable=trials()
    for trial in unstable:
        trial["history"][-1]["validation"]["macro_f1_supported"]=.5
    with pytest.raises(ValueError,match="eligible"): _choose_protocol(unstable,policy()["pilot"])


def test_public_selection_rejects_unverified_metric_dicts():
    from src.e3.campaign import select_common_protocol,validate_completed_head_run
    from src.e3.config import policy
    with pytest.raises(ValueError): select_common_protocol(trials(),policy()["pilot"])


def test_rehashed_incomplete_completed_run_is_not_complete(tmp_path):
    from src.e3.campaign import validate_completed_head_run
    from src.e3.io import atomic_json
    atomic_json(tmp_path/"complete.json",{"kind":"e3_completed_head_run_v1","stage":"final","run_id":"fake"})
    with pytest.raises(ValueError): validate_completed_head_run(tmp_path)


def test_final_matrix_requires_twelve_shared_profiles_and_paired_initial_states():
    from src.e3.campaign import validate_final_declarations
    with pytest.raises(ValueError): validate_final_declarations([])
    with pytest.raises(ValueError): validate_final_declarations([{"family":"E1","source_seed":2001,"arm":"natural"}]*12)


def test_positive_final_matrix_then_pair_or_common_profile_drift():
    from src.e3.campaign import validate_final_declarations
    from src.e3.config import registry,policy,fingerprint
    from src.e1.data import derived_seed
    from tests.e3.helpers import declaration,digest
    records=[]
    for family in ("E1","E2"):
        for seed in (2001,3001,4001):
            ref=next(r for r in registry()["selected_sources"] if (r["family"],r["source_seed"])==(family,seed))
            for arm in ("natural","crt"):
                d=declaration(family,arm)
                d.update(kind="e3_final_head_run_v1",stage="final",source_seed=seed,root_seed=seed,
                         head_freeze_receipt_sha256=digest("no actual receipt exists"),required_authorization_gate="E3-FINAL-AUTH")
                d["source"].update(source_seed=seed,selected_epoch=ref["selected_epoch"],binary_sha256=ref["required_binary_sha256"])
                d["initialization"].update(initialization_seed=derived_seed(seed,"E3-v1","final",family,seed,"head-init"),
                    initializer_policy_sha256=fingerprint(policy()["initializer"]),tensor_state_sha256=digest(family+str(seed)))
                d["run_id"]="e3-final-"+fingerprint({k:v for k,v in d.items() if k!="run_id"})[:24]
                records.append(d)
    assert len(validate_final_declarations(records))==12
    broken=copy.deepcopy(records); broken[1]["initialization"]["tensor_state_sha256"]=digest("different initial state")
    broken[1]["run_id"]="e3-final-"+fingerprint({k:v for k,v in broken[1].items() if k!="run_id"})[:24]
    with pytest.raises(ValueError,match="pair"): validate_final_declarations(broken)

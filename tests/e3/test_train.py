import copy
import pytest


def test_fixed_budget_runs_every_window_despite_decreasing_validation():
    from src.e3.train import _execute_windows
    history=[]; trained=[]; scores=iter([.5,.4,.3])
    progress={"window":0,"presentations":0,"updates":0,"history":[],"selected":None,"selected_weights":None}
    def train():
        trained.append(1); return {"presentations":5,"updates":2,"ce":1.}
    result=_execute_windows(budget={"windows":3,"presentations_per_window":5,"updates_per_window":2},progress=progress,
        train=train,evaluate=lambda:{"macro_f1_supported":next(scores)},weights=lambda:{"weight":"fixture"},
        commit=lambda p:history.append(copy.deepcopy(p)),stop_requested=lambda:False)
    assert len(trained)==3 and result["updates"]==6 and result["presentations"]==15
    assert result["selected"]["window"]==1 and len(history)==3


def test_completed_budget_publication_recovery_never_trains_again():
    from src.e3.train import _execute_windows
    progress={"window":3,"presentations":15,"updates":6,"history":[1,2,3],"selected":{"window":1},"selected_weights":{}}
    fail=lambda:pytest.fail("terminal recovery must not train or evaluate")
    result=_execute_windows(budget={"windows":3,"presentations_per_window":5,"updates_per_window":2},progress=progress,
        train=fail,evaluate=fail,weights=fail,commit=lambda p:pytest.fail("no new boundary"),stop_requested=lambda:False)
    assert result==progress

"""WorkflowRunOutput/TeamRunOutput.from_dict must restore the status enum.

Stored runs carry the status as its string value: ``to_dict()`` writes
``status.value`` and every DB adapter persists that dict. Readers such as the
workflow WebSocket DB-replay path evaluate ``run.status.value`` on the
deserialized run, so ``from_dict`` must hand back a ``RunStatus`` - exactly what
``RunOutput.from_dict`` already does for agent runs.
"""

import json

import pytest

from agno.run.base import RunStatus
from agno.run.team import TeamRunOutput
from agno.run.workflow import WorkflowRunOutput

OUTPUTS = [
    pytest.param(WorkflowRunOutput, {"workflow_id": "wf1"}, id="workflow"),
    pytest.param(TeamRunOutput, {"team_id": "team1"}, id="team"),
]


@pytest.mark.parametrize(("output_cls", "extra"), OUTPUTS)
def test_from_dict_returns_the_status_enum(output_cls, extra):
    original = output_cls(run_id="r1", status=RunStatus.completed, **extra)

    restored = output_cls.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.status is RunStatus.completed
    assert isinstance(restored.status, RunStatus)
    assert restored.status.value == "COMPLETED"


@pytest.mark.parametrize(("output_cls", "extra"), OUTPUTS)
def test_consumer_reads_the_status_value_after_a_db_style_round_trip(output_cls, extra):
    original = output_cls(run_id="r1", status=RunStatus.completed, **extra)
    restored = output_cls.from_dict(json.loads(json.dumps(original.to_dict())))

    # The workflow WebSocket DB-replay branch (os/routers/workflows/router.py)
    # evaluates exactly this expression on a run loaded from the database.
    status_value = restored.status.value if restored.status else "unknown"

    assert status_value == "COMPLETED"


@pytest.mark.parametrize(("output_cls", "extra"), OUTPUTS)
def test_from_dict_keeps_an_unknown_status_string(output_cls, extra):
    restored = output_cls.from_dict({"run_id": "r1", "status": "SOMETHING_NEW", **extra})

    assert restored.status == "SOMETHING_NEW"

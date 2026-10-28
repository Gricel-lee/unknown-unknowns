"""Checks on unknown_unknowns.controller_generator."""
from pathlib import Path

import pytest

from unknown_unknowns.controller_generator.controller import Controller
from unknown_unknowns.controller_generator import PrismController
from unknown_unknowns.controller_generator import DEFAULT_PRISM

EXAMPLES = Path(__file__).parent.parent / "examples"
ROBOT, CONSENSUS, FX, UAV = (EXAMPLES / name / "prism_templates" for name in ("robot", "consensus", "fx", "uav"))

def costs(path1):
    return {"path1": path1, "path2": 30.0}


# path2 always costs 30; path1 is cheaper when light=0 and dearer when light=1 (common)
SAMPLES = ([({"light": 0, "wet": 0}, costs(10.0))] * 2 + [({"light": 0, "wet": 1}, costs(20.0))] * 2
           + [({"light": 1, "wet": 0}, costs(40.0))] * 4 + [({"light": 1, "wet": 1}, costs(50.0))] * 2)


def test_controller_is_abstract():
    with pytest.raises(TypeError):
        Controller()


def test_constants_no_unknowns(tmp_path):
    consts = PrismController(tmp_path, ROBOT, data=SAMPLES).constants()
    assert consts == {"c1": 32.0, "c2": 30.0}   # mean cost of each path


def test_constants_two_unknowns(tmp_path):
    consts = PrismController(tmp_path, ROBOT, data=SAMPLES, unknowns=["light", "wet"]).constants()
    assert consts["p00"] == 0.2 and consts["p10"] == 0.4 and consts["p01"] == 0.2
    assert consts["c1_10"] == 40.0 and consts["c2_10"] == 30.0


def test_non_binary_unknown(tmp_path):
    data = SAMPLES + [({"light": 2, "wet": 0}, costs(1.0))]
    with pytest.raises(ValueError, match="0 and 1"):
        PrismController(tmp_path, ROBOT, data=data, unknowns=["light"]).constants()


@pytest.mark.skipif(not DEFAULT_PRISM.exists(), reason="PRISM not installed")
def test_rewards_with_prism(tmp_path):
    without_u = PrismController(tmp_path, ROBOT, data=SAMPLES, name="without")
    assert without_u.get_controller() == ["action2"]   # path2 is cheaper on average
    assert without_u.get_metrics()["reward"] == pytest.approx(30.0)   # min(32, 30)
    with_u = PrismController(tmp_path, ROBOT, data=SAMPLES, unknowns=["light"], name="with")
    # light=0: path1 (15 < 30); light=1: path2 (43.3 > 30)
    assert with_u.get_controller() == ["action1", "action2"]
    assert with_u.environments() == [{"light": 0}, {"light": 1}]
    assert with_u.get_metrics()["reward"] == pytest.approx(0.4 * 15 + 0.6 * 30)


def test_model_names_unknowns(tmp_path):
    text = PrismController(tmp_path, ROBOT, data=SAMPLES, unknowns=["light", "wet"]
                           ).generate_prism_file_from_unknowns().read_text()
    assert "(u1=light, u2=wet)" in text
    assert "// pXY = probability of light=X, wet=Y" in text


@pytest.mark.skipif(not DEFAULT_PRISM.exists(), reason="PRISM not installed")
def test_consensus_controller(tmp_path):
    # server1 is slow when busy=1 (half the samples), server2 always takes 20 ms
    data = ([({"busy": 0}, {"server1": 10.0, "server2": 20.0})] * 5
            + [({"busy": 1}, {"server1": 40.0, "server2": 20.0})] * 5)
    without_u = PrismController(tmp_path, CONSENSUS, data=data, name="without")
    assert without_u.get_controller() == ["flip2"]                 # 25 vs 20 on average
    assert without_u.get_metrics()["reward"] == pytest.approx(4 * 20)   # 4 flips on average
    with_u = PrismController(tmp_path, CONSENSUS, data=data, unknowns=["busy"], name="with")
    assert with_u.get_controller() == ["flip1", "flip2"]
    assert with_u.get_metrics()["reward"] == pytest.approx(4 * (0.5 * 10 + 0.5 * 20))


@pytest.mark.skipif(not DEFAULT_PRISM.exists(), reason="PRISM not installed")
def test_fx_controller(tmp_path):
    # technical analysis is slow on volatile days (half the samples), fundamental always 20
    data = ([({"volatile": 0}, {"technical": 5.0, "fundamental": 20.0})] * 5
            + [({"volatile": 1}, {"technical": 60.0, "fundamental": 20.0})] * 5)
    without_u = PrismController(tmp_path, FX, data=data, name="without")
    with_u = PrismController(tmp_path, FX, data=data, unknowns=["volatile"], name="with")
    assert without_u.get_controller() == ["fundanalysis"]          # 32.5 (+ market watch) vs 20
    assert with_u.get_controller() == ["marketwatch", "fundanalysis"]
    assert with_u.get_metrics()["reward"] < without_u.get_metrics()["reward"]


@pytest.mark.skipif(not DEFAULT_PRISM.exists(), reason="PRISM not installed")
def test_uav_controller(tmp_path):
    # the operator is slow when busy (half the samples); onboard always takes 14
    data = ([({"operator_busy": 0}, {"operator": 8.0, "onboard": 14.0})] * 5
            + [({"operator_busy": 1}, {"operator": 30.0, "onboard": 14.0})] * 5)
    with_u = PrismController(tmp_path, UAV, data=data, unknowns=["operator_busy"], name="with")
    assert with_u.get_controller() == ["ask_operator", "onboard"]
    assert with_u.get_metrics()["reward"] > 0


def test_costs_follow_output_order(tmp_path):
    # c1 is the first output of each sample, c2 the second (the order in context_distributions.txt)
    data = [({"light": 0}, {"slow": 9.0, "fast": 1.0})] * 4
    assert PrismController(tmp_path, ROBOT, data=data).constants() == {"c1": 9.0, "c2": 1.0}


def test_action_costs_override(tmp_path):
    data = [({"light": 0}, {"path1": 10.0, "path2": 20.0})] * 4
    costs = {1: lambda v: v["path1"] + v["path2"], 2: lambda v: 5.0}
    assert PrismController(tmp_path, ROBOT, data=data, action_costs=costs).constants() == {"c1": 30.0, "c2": 5.0}


def test_single_value_needs_action_costs(tmp_path):
    with pytest.raises(ValueError, match="action_costs"):
        PrismController(tmp_path, ROBOT, data=[({"light": 0}, 3.0)] * 4).constants()

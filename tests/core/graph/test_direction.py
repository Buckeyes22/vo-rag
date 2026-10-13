from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vo_rag.core.graph.direction import orient_edges, orient_low_confidence_via_llm
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.core.llm.response import LLMResponse

_NOW = datetime.now(UTC)


def _node(label: str, node_type: GraphNodeType = GraphNodeType.CONCEPT) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=node_type, canonical_label=label),
        node_type=node_type,
        label=label,
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-1"],
        created_at=_NOW,
    )


def _edge(source: GraphNode, target: GraphNode, edge_type: GraphEdgeType) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type=edge_type,
        ),
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=edge_type,
        book_ids=["book-1"],
        evidence_chunk_ids=["chunk-1"],
        weight=1.0,
        created_at=_NOW,
    )


def _labels_for(edge: GraphEdge, nodes: list[GraphNode]) -> tuple[str, str]:
    labels = {node.node_id: node.label for node in nodes}
    return labels[edge.source_node_id], labels[edge.target_node_id]


def test_containment_orients_is_a_with_longer_label_as_source() -> None:
    generic_generator = _node("Generator", GraphNodeType.COMPONENT)
    specific_generator = _node("Honda EU2000i Generator", GraphNodeType.COMPONENT)
    fia_harness = _node("FIA Harness", GraphNodeType.COMPONENT)
    harness = _node("Harness", GraphNodeType.COMPONENT)
    nodes = [generic_generator, specific_generator, fia_harness, harness]
    reversed_edge = _edge(generic_generator, specific_generator, GraphEdgeType.IS_A)
    already_correct = _edge(fia_harness, harness, GraphEdgeType.IS_A)

    oriented, coverage = orient_edges(nodes, [reversed_edge, already_correct])

    assert _labels_for(oriented[0], nodes) == ("Honda EU2000i Generator", "Generator")
    assert oriented[0].edge_id == reversed_edge.edge_id
    assert _labels_for(oriented[1], nodes) == ("FIA Harness", "Harness")
    assert coverage.n_directed == 2
    assert coverage.n_high == 2
    assert coverage.n_medium == 0
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0
    assert coverage.low_confidence_edge_ids == []


def test_head_noun_match_orients_compound_as_high_confidence_child() -> None:
    pads = _node("Pads", GraphNodeType.COMPONENT)
    g_lok_pads = _node("G-Lok Pads", GraphNodeType.COMPONENT)
    racing_seat = _node("Racing Seat", GraphNodeType.COMPONENT)
    cobra_seat = _node("Cobra Evolution Seat", GraphNodeType.COMPONENT)
    nodes = [pads, g_lok_pads, racing_seat, cobra_seat]
    single_head_first = _edge(pads, g_lok_pads, GraphEdgeType.IS_A)
    less_qualified_first = _edge(racing_seat, cobra_seat, GraphEdgeType.PART_OF)

    oriented, coverage = orient_edges(nodes, [single_head_first, less_qualified_first])

    assert _labels_for(oriented[0], nodes) == ("G-Lok Pads", "Pads")
    assert oriented[0].edge_id == single_head_first.edge_id
    assert oriented[0].edge_type == GraphEdgeType.IS_A
    assert _labels_for(oriented[1], nodes) == ("Cobra Evolution Seat", "Racing Seat")
    assert oriented[1].edge_id == less_qualified_first.edge_id
    assert oriented[1].edge_type == GraphEdgeType.PART_OF
    assert coverage.n_directed == 2
    assert coverage.n_high == 2
    assert coverage.n_medium == 0
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0
    assert coverage.low_confidence_edge_ids == []


def test_part_of_falls_back_to_parent_degree_and_swaps_whole_first_edge() -> None:
    engine = _node("Engine", GraphNodeType.COMPONENT)
    intake = _node("Intake", GraphNodeType.COMPONENT)
    spark_plug = _node("Spark Plug", GraphNodeType.COMPONENT)
    block = _node("Block", GraphNodeType.COMPONENT)
    nodes = [engine, intake, spark_plug, block]
    reversed_edge = _edge(engine, intake, GraphEdgeType.PART_OF)
    supporting_edges = [
        _edge(spark_plug, engine, GraphEdgeType.PART_OF),
        _edge(block, engine, GraphEdgeType.PART_OF),
    ]

    oriented, coverage = orient_edges(nodes, [reversed_edge, *supporting_edges])

    assert _labels_for(oriented[0], nodes) == ("Intake", "Engine")
    assert coverage.n_directed == 3
    assert coverage.n_high == 0
    assert coverage.n_medium == 3
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0


def test_parent_degree_uses_is_a_and_part_of_target_frequency_for_medium_confidence() -> None:
    vehicle_dynamics = _node("Vehicle Dynamics")
    yaw_control = _node("Yaw Control")
    load_transfer = _node("Load Transfer")
    tyre_grip = _node("Tyre Grip")
    nodes = [vehicle_dynamics, yaw_control, load_transfer, tyre_grip]
    reversed_edge = _edge(vehicle_dynamics, yaw_control, GraphEdgeType.IS_A)
    supporting_edges = [
        _edge(load_transfer, vehicle_dynamics, GraphEdgeType.IS_A),
        _edge(tyre_grip, vehicle_dynamics, GraphEdgeType.PART_OF),
    ]

    oriented, coverage = orient_edges(nodes, [reversed_edge, *supporting_edges])

    assert _labels_for(oriented[0], nodes) == ("Yaw Control", "Vehicle Dynamics")
    assert coverage.n_directed == 3
    assert coverage.n_medium == 3
    assert coverage.coverage == 1.0


def test_used_by_type_signal_swaps_process_to_tool_and_keeps_tool_to_process() -> None:
    powder_coating = _node("Powder Coating", GraphNodeType.CONCEPT)
    wheels = _node("Wheels", GraphNodeType.COMPONENT)
    nodes = [powder_coating, wheels]
    reversed_edge = _edge(powder_coating, wheels, GraphEdgeType.USED_BY)
    already_correct = _edge(wheels, powder_coating, GraphEdgeType.USED_BY)

    oriented, coverage = orient_edges(nodes, [reversed_edge, already_correct])

    assert _labels_for(oriented[0], nodes) == ("Wheels", "Powder Coating")
    assert _labels_for(oriented[1], nodes) == ("Wheels", "Powder Coating")
    assert coverage.n_directed == 2
    assert coverage.n_high == 0
    assert coverage.n_medium == 2
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0


def test_used_by_process_phrase_signal_orients_tool_to_process_when_types_are_ambiguous() -> None:
    adjusting_oversteer = _node("Adjusting Oversteer", GraphNodeType.CONCEPT)
    sway_bar = _node("Sway Bar", GraphNodeType.CONCEPT)
    pyrometer = _node("Pyrometer", GraphNodeType.CONCEPT)
    tire_measurement = _node("Tire Temperature Measurement", GraphNodeType.CONCEPT)
    nodes = [adjusting_oversteer, sway_bar, pyrometer, tire_measurement]
    process_first = _edge(adjusting_oversteer, sway_bar, GraphEdgeType.USED_BY)
    tool_first = _edge(pyrometer, tire_measurement, GraphEdgeType.USED_BY)

    oriented, coverage = orient_edges(nodes, [process_first, tool_first])

    assert _labels_for(oriented[0], nodes) == ("Sway Bar", "Adjusting Oversteer")
    assert oriented[0].edge_id == process_first.edge_id
    assert _labels_for(oriented[1], nodes) == ("Pyrometer", "Tire Temperature Measurement")
    assert oriented[1].edge_id == tool_first.edge_id
    assert coverage.n_directed == 2
    assert coverage.n_high == 0
    assert coverage.n_medium == 2
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0


def test_academic_process_suffixes_orient_concept_method_used_by_edges() -> None:
    tire_forces = _node("Tire Forces", GraphNodeType.CONCEPT)
    corner_estimation = _node("Corner-Based Estimation", GraphNodeType.CONCEPT)
    parameter_identification = _node("Parameter Identification", GraphNodeType.METHOD)
    recursive_least_squares = _node("Recursive Least Squares", GraphNodeType.CONCEPT)
    sideslip_angle = _node("Vehicle Sideslip Angle", GraphNodeType.CONCEPT)
    sliding_mode_observer = _node("Sliding Mode Observer", GraphNodeType.METHOD)
    nodes = [
        tire_forces,
        corner_estimation,
        parameter_identification,
        recursive_least_squares,
        sideslip_angle,
        sliding_mode_observer,
    ]
    already_tool_to_process = _edge(tire_forces, corner_estimation, GraphEdgeType.USED_BY)
    process_first = _edge(
        parameter_identification,
        recursive_least_squares,
        GraphEdgeType.USED_BY,
    )
    academic_observer = _edge(sideslip_angle, sliding_mode_observer, GraphEdgeType.USED_BY)

    oriented, coverage = orient_edges(
        nodes,
        [already_tool_to_process, process_first, academic_observer],
    )

    assert _labels_for(oriented[0], nodes) == ("Tire Forces", "Corner-Based Estimation")
    assert _labels_for(oriented[1], nodes) == (
        "Recursive Least Squares",
        "Parameter Identification",
    )
    assert _labels_for(oriented[2], nodes) == ("Vehicle Sideslip Angle", "Sliding Mode Observer")
    assert coverage.n_directed == 3
    assert coverage.n_high == 0
    assert coverage.n_medium == 3
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0
    assert coverage.low_confidence_edge_ids == []


def test_category_head_signal_orients_specific_item_to_category_at_medium_confidence() -> None:
    spare_parts = _node("Spare Parts", GraphNodeType.CONCEPT)
    heater_hose = _node("Heater Hose", GraphNodeType.CONCEPT)
    pump_kit = _node("Pump Kit", GraphNodeType.CONCEPT)
    fuel_pump = _node("Fuel Pump", GraphNodeType.CONCEPT)
    nodes = [spare_parts, heater_hose, pump_kit, fuel_pump]
    category_first = _edge(spare_parts, heater_hose, GraphEdgeType.PART_OF)
    specific_first = _edge(fuel_pump, pump_kit, GraphEdgeType.IS_A)

    oriented, coverage = orient_edges(nodes, [category_first, specific_first])

    assert _labels_for(oriented[0], nodes) == ("Heater Hose", "Spare Parts")
    assert oriented[0].edge_id == category_first.edge_id
    assert oriented[0].edge_type == GraphEdgeType.PART_OF
    assert _labels_for(oriented[1], nodes) == ("Fuel Pump", "Pump Kit")
    assert oriented[1].edge_id == specific_first.edge_id
    assert oriented[1].edge_type == GraphEdgeType.IS_A
    assert coverage.n_directed == 2
    assert coverage.n_high == 0
    assert coverage.n_medium == 2
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0


def test_non_directed_edges_pass_through_unchanged_and_do_not_count() -> None:
    aero = _node("Aero")
    drag = _node("Drag")
    nodes = [aero, drag]
    relates = _edge(aero, drag, GraphEdgeType.RELATES_TO)
    mentions = _edge(aero, drag, GraphEdgeType.MENTIONS)

    oriented, coverage = orient_edges(nodes, [relates, mentions])

    assert oriented == [relates, mentions]
    assert coverage.n_directed == 0
    assert coverage.n_high == 0
    assert coverage.n_medium == 0
    assert coverage.n_low == 0
    assert coverage.coverage == 1.0
    assert coverage.low_confidence_edge_ids == []


def test_coverage_math_tracks_high_medium_low_and_low_confidence_edge_ids() -> None:
    generator = _node("Generator", GraphNodeType.COMPONENT)
    honda_generator = _node("Honda EU2000i Generator", GraphNodeType.COMPONENT)
    powder_coating = _node("Powder Coating", GraphNodeType.CONCEPT)
    wheels = _node("Wheels", GraphNodeType.COMPONENT)
    curing = _node("Curing", GraphNodeType.METHOD)
    nodes = [generator, honda_generator, powder_coating, wheels, curing]
    high = _edge(generator, honda_generator, GraphEdgeType.IS_A)
    medium = _edge(powder_coating, wheels, GraphEdgeType.USED_BY)
    low = _edge(powder_coating, curing, GraphEdgeType.USED_BY)

    _oriented, coverage = orient_edges(nodes, [high, medium, low])

    assert coverage.n_directed == 3
    assert coverage.n_high == 1
    assert coverage.n_medium == 1
    assert coverage.n_low == 1
    assert coverage.coverage == pytest.approx(2 / 3)
    assert coverage.low_confidence_edge_ids == [low.edge_id]


def test_new_signals_raise_coverage_for_edges_that_were_previously_low_confidence() -> None:
    pads = _node("Pads", GraphNodeType.CONCEPT)
    g_lok_pads = _node("G-Lok Pads", GraphNodeType.CONCEPT)
    oversteer = _node("Adjusting Oversteer", GraphNodeType.CONCEPT)
    sway_bar = _node("Sway Bar", GraphNodeType.CONCEPT)
    spare_parts = _node("Spare Parts", GraphNodeType.CONCEPT)
    heater_hose = _node("Heater Hose", GraphNodeType.CONCEPT)
    baseline = _node("Baseline", GraphNodeType.CONCEPT)
    reference = _node("Reference", GraphNodeType.CONCEPT)
    nodes = [
        pads,
        g_lok_pads,
        oversteer,
        sway_bar,
        spare_parts,
        heater_hose,
        baseline,
        reference,
    ]
    head_noun = _edge(pads, g_lok_pads, GraphEdgeType.IS_A)
    process_phrase = _edge(oversteer, sway_bar, GraphEdgeType.USED_BY)
    category_head = _edge(spare_parts, heater_hose, GraphEdgeType.PART_OF)
    still_low = _edge(baseline, reference, GraphEdgeType.USED_BY)

    oriented, coverage = orient_edges(
        nodes,
        [head_noun, process_phrase, category_head, still_low],
    )

    assert _labels_for(oriented[0], nodes) == ("G-Lok Pads", "Pads")
    assert _labels_for(oriented[1], nodes) == ("Sway Bar", "Adjusting Oversteer")
    assert _labels_for(oriented[2], nodes) == ("Heater Hose", "Spare Parts")
    assert _labels_for(oriented[3], nodes) == ("Baseline", "Reference")
    assert coverage.n_directed == 4
    assert coverage.n_high == 1
    assert coverage.n_medium == 2
    assert coverage.n_low == 1
    assert coverage.coverage == pytest.approx(3 / 4)
    assert coverage.low_confidence_edge_ids == [still_low.edge_id]


def test_orient_edges_is_deterministic() -> None:
    parent = _node("Race Car")
    child = _node("Chassis")
    tool = _node("Torque Wrench", GraphNodeType.COMPONENT)
    process = _node("Wheel Nut Tightening", GraphNodeType.METHOD)
    nodes = [parent, child, tool, process]
    edges = [
        _edge(parent, child, GraphEdgeType.PART_OF),
        _edge(process, tool, GraphEdgeType.USED_BY),
        _edge(parent, process, GraphEdgeType.RELATES_TO),
    ]

    first_edges, first_coverage = orient_edges(nodes, edges)
    second_edges, second_coverage = orient_edges(nodes, edges)

    assert first_edges == second_edges
    assert first_coverage == second_coverage


@pytest.mark.asyncio
async def test_orient_low_confidence_via_llm_swaps_flagged_edge_only() -> None:
    system = _node("Suspension System", GraphNodeType.COMPONENT)
    spring = _node("Spring", GraphNodeType.COMPONENT)
    tire_probe = _node("Pyrometer", GraphNodeType.COMPONENT)
    tire_check = _node("Tire Temperature Check", GraphNodeType.METHOD)
    nodes = [system, spring, tire_probe, tire_check]
    flagged = _edge(system, spring, GraphEdgeType.PART_OF)
    unflagged = _edge(tire_probe, tire_check, GraphEdgeType.USED_BY)
    seen_user_prompt = ""

    async def fake_llm_call(
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        nonlocal seen_user_prompt
        seen_user_prompt = user_prompt
        assert "part_of: source is a part of target" in user_prompt
        assert "used_by: source is a tool used by target" in user_prompt
        assert '1: "Suspension System"  --part_of--> "Spring"' in user_prompt
        assert "technical graph edge direction" in system_prompt
        assert max_tokens > 0
        assert temperature == 0.0
        return LLMResponse(
            text=(
                '{"items": [{"i": 1, "source_label": "Spring", '
                '"target_label": "Suspension System"}]}'
            ),
            model="fake-27b",
        )

    corrected = await orient_low_confidence_via_llm(
        nodes,
        [flagged, unflagged],
        [flagged.edge_id],
        fake_llm_call,
    )

    assert seen_user_prompt
    assert _labels_for(corrected[0], nodes) == ("Spring", "Suspension System")
    assert corrected[0].edge_id == flagged.edge_id
    assert corrected[1] == unflagged


@pytest.mark.asyncio
async def test_orient_low_confidence_via_llm_parse_failure_leaves_edges_unchanged() -> None:
    parent = _node("Engine", GraphNodeType.COMPONENT)
    child = _node("Intake", GraphNodeType.COMPONENT)
    edge = _edge(parent, child, GraphEdgeType.PART_OF)

    async def broken_llm_call(
        _system_prompt: str,
        _user_prompt: str,
        _max_tokens: int,
        _temperature: float,
    ) -> LLMResponse:
        return LLMResponse(text="not json", model="fake-27b")

    corrected = await orient_low_confidence_via_llm(
        [parent, child],
        [edge],
        [edge.edge_id],
        broken_llm_call,
    )

    assert corrected == [edge]

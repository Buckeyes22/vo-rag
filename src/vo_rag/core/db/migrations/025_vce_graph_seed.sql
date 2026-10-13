-- 025_vce_graph_seed.sql
-- Seeds initial coaching_graph_nodes from controlled_vocab.yaml.
-- Operator may extend the YAML; subsequent migrations carry deltas.

INSERT INTO vce.coaching_graph_nodes (node_type, canonical_label, aliases, description) VALUES
  ('principle', 'brake_release_smoothness', ARRAY['smooth brake release', 'progressive brake off'], 'Releasing brake pressure smoothly to maintain stable platform and weight transfer'),
  ('principle', 'entry_speed_management', ARRAY['entry speed', 'corner entry speed'], 'Controlling speed at corner entry to enable optimal apex line'),
  ('principle', 'apex_offset', ARRAY['apex point', 'apex location'], 'Choosing apex point that optimizes exit line and minimum speed'),
  ('principle', 'throttle_application_progressive', ARRAY['progressive throttle', 'smooth throttle'], 'Applying throttle progressively at corner exit'),
  ('principle', 'line_optimization', ARRAY['racing line', 'ideal line'], 'Choosing line that minimizes lap time across the whole corner'),
  ('principle', 'late_apex_priority', ARRAY['late apex'], 'Sacrificing entry speed for better exit line'),
  ('principle', 'trail_brake_distance', ARRAY['trail braking', 'trail brake'], 'How far into the corner to carry brake pressure'),
  ('principle', 'weight_transfer', ARRAY['load transfer', 'weight management'], 'Managing longitudinal and lateral load transfer through driver inputs'),
  ('principle', 'exit_priority', ARRAY['exit first'], 'Prioritizing corner exit speed over entry speed'),
  ('principle', 'reference_lap_comparison_discipline', ARRAY['compare to reference', 'delta to reference'], 'Systematically comparing current lap inputs to a reference lap'),
  ('pattern', 'overdrive_into_corner', ARRAY['over-driving entry', 'too hot into the corner'], 'Carrying too much speed at corner entry, compromising the rest of the corner'),
  ('pattern', 'early_throttle', ARRAY['throttle too early', 'premature throttle'], 'Applying throttle before the apex, forcing a wider exit line'),
  ('pattern', 'late_brake_release', ARRAY['brake too long', 'still braking at apex'], 'Holding brake too long into the corner, slowing the car beyond optimal'),
  ('pattern', 'under_braking_too_long', ARRAY['coasting between brake and throttle'], 'Time gap between brake release and throttle application'),
  ('pattern', 'over_apex_too_early', ARRAY['early apex'], 'Reaching the apex point too early, compromising exit'),
  ('drill', 'brake_zone_marker', NULL, 'Identifying a fixed reference point for brake initiation'),
  ('drill', 'slow_in_fast_out', NULL, 'Deliberately slowing entry to allow earlier throttle application'),
  ('drill', 'reference_lap_comparison', NULL, 'Comparing current lap data to a saved reference lap'),
  ('drill', 'single_corner_loop', NULL, 'Repeating one corner in isolation to dial in technique')
ON CONFLICT (node_type, canonical_label) DO NOTHING;

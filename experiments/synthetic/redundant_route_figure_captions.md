# Dissertation-ready captions: redundant route methodology

## Figure: Factorial Functional carriage on a redundant two-route task

The target is available through two correct routes: an exact local copy and a
distant record selected jointly by a semantic key and a structural bank. (a)
The local route is sufficient, while dense attention can instantiate the
distant lookup over carriers at distances 1, 4, 4, and 6. (b) Existing
semantic and structural donor interventions are extended by one joint
intervention. The second-order finite contrast is projected onto the clean
output gradient and aggregated by carrier distance to give semantic-structural
interaction carriage. (c) The resulting quantities answer separate questions.
An exact local-only refit establishes that the distant route is not necessary
for the task; interaction mass measures how strongly the trained model
instantiates the conjunctive route; and route-specific rescue and damage test
the corresponding backup capacity and exposure. Usage and reliance are not
interpreted as task necessity.

## Figure: Conditional carriage recovers redundant conjunctive route allocation

Top: controlled route allocation. (a) Raw interaction carriage increases with
the known distant-route mixture, including for a mixture learned from clean
data, whereas the carrier-normalised expected distance remains effectively
constant. The primary signal is therefore the amount of interaction carriage,
not a shift in its fixed geometry. (b) Interaction mass predicts both rescue
when the redundant local copy fails and damage when the selected distant record
fails. These are the two behavioural consequences of the same instantiated
route. (c) Removing the distant route from the frozen model becomes increasingly
damaging, but a model refitted using only the local copy remains exact. Frozen
model reliance is therefore separated from task-level sufficiency.

Bottom: emergent allocation in a two-layer, four-head transformer without an
explicit local/global gate. Distant-record corruption is fixed at 5%, while
local-copy corruption is varied during training. (d) Clean interaction carriage
increases as the local route becomes less reliable. (e) Across 16 independently
trained condition-seed models, interaction mass predicts local-failure rescue
(`r = 0.992`) and distant-failure damage (`r = 0.994`). (f) The across-head
semantic and structural score profiles increasingly align while their relative
imbalance falls. In this positive control, score co-peaking reflects a
conjunctive lookup in which the same distant route requires both the semantic
key and structural bank. Small points are individual seeds; connected large
points are condition means and error bars show one standard deviation across
four seeds.

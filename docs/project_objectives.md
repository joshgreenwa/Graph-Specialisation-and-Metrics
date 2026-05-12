# Project Objectives

The project studies how graph transformer architectures encode and route structural and symbolic information.

Key questions:

- Do different architectures develop different forms of attention-head specialisation?
- Which heads or layers are most sensitive to symbolic labels, graph distance, local structure, or global graph context?
- How stable are specialisation patterns across tasks, checkpoints, random seeds, and model families?
- Can specialisation metrics explain differences in downstream performance?

Initial model families:

- Graphormer
- GraphGPS
- GRIT
- CSA
- Exphormer

Initial task component:

- ZINC molecular regression training runs.

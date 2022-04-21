# VL-BERT-Graph

A visual-linguistic model that uses GNNs for improving input and output representations. Project for EECE 571F.

### Notable changes: 

1. You can view the model configurations with all hyperparameters at [/cfgs/refcoco](/cfgs/refcoco).
2. The dataloader is updated to perform edge feature construction. See lines 253 onwards in [refcoco.py](/refcoco/data/datasets/refcoco.py) for more details.
3. GNN is implemented here: [gnn.py](/common/gnn.py). An alternate GNN model is implemented in [gnnV2.py](/common/gnnV2.py).
4. The model is updated to use the GNN and handle different GNN types. See lines 131 onwards in [visual_linguistic_bert.py](/common/visual_linguistic_bert.py)

### Setup

Use the instructions provided in [README-OLD.md](/README-OLD.md).
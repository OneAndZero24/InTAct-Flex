## Methods
Continual learning methods.
- `composer.py` - Main `Composer` class implements naive joint training flow, can be extended via plugins.
- `method_plugin_abc.py` - `MethodABC` base class.
- `regularization.py` - Loss components used in different methods for regularization.
- `lwf.py` - Learning without Forgetting.
- `ewc.py` - Elastic Weight Consolidation.
- `mas.py` - Memory Aware Synapses.
- `si.py` - Synaptic Intelligence.
- `interval_penalization_big_model.py` - InTAct implementation for a Vision Transformer (ViT).
- `interval_penalization_mlp.py` - InTAct implementation for a multilayer perceptron (MLP).
- `interval_penalization_resnet18_cls.py` - InTAct implementation for ResNet-18, where InTAct is applied to the classifier only.
- `interval_penalization_resnet18_last_block.py` - InTAct implementation for ResNet-18, where the last block is unfrozen and protected by InTAct.

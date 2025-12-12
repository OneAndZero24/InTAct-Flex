## Models
Full models and custom layers.

- `inc_classifier.py` - `IncrementalClassifier` automatically extends the given layer to handle new classes in CIL; used as the head in each model.
- `cl_module_abc.py` - `CLModuleABC` serves as a base class for continual learning modules.
- `mlp.py` - Customizable MLP wrapper.
- `big_model.py` - "Big model": pretrained torchvision backbone with a custom head.
- `resnet18_interval_cls.py` - ResNet18 with InTAct, where only the classifier is protected by InTAct.
- `resnet18_interval_last_block.py` - ResNet18 with InTAct, where the last convolutional block is unfrozen and protected by InTAct.
- `resnet18.py` - Standard ResNet18 architecture.
- `layer/` - Custom layer implementations. Use the `instantiate` & `instantiate2D` APIs.
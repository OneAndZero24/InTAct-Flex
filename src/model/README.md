## Models
Full models and custom layers.
- `inc_classifier.py` - `IncerementalClassifier` automatically extends given layer to handle new classes in CIL, used as head in each model
- `cl_module_abc.py` - `CLModuleABC` acts as a base class for CL Modules that record activations
- `mlp.py` - Customizable MLP wrapper
- `big_model.py` - "Big model" - pretrained torchvision backbone with custom head
- `resnet18_interval.py` - ResNet18 with interval-based training, used for CIL
- `layer/` - Custom layer implementations. Use `instantiate` & `instantiate2D` API

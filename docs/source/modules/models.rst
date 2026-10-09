Models
======

Model implementations live in ``blades/models``; backbone definitions are in
``blades/models/backbones``. Model construction and task configuration connect
these definitions to the training runtime.

For a new or modified architecture, edit the local model file and update the
experiment's task/model configuration. Match the model's input shape and output
classes to its dataset. The synthetic CPU example in
``examples/smoke_train.py`` provides a small model and dataset combination for
checking the training path.

# Multimodal signals

The multimodal layer accepts bounded, confidence-weighted observations from text/check-ins and optional voice features. It intentionally does not map a signal to a diagnosis. The current voice path is a feature contract; a production speech encoder must be evaluated separately and plugged into `VoiceSignals`.

Every modality contributes only according to its confidence, and missing modalities are valid.

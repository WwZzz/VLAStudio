# FastWAM

Inference adapter for the official FastWAM Base LIBERO checkpoint.
Run `examples/08_fastwam_libero/setup.sh`, activate its environment, then use
`vla.load_policy("fastwam", checkpoint="/path/to/bundle")` for evaluation or serving.
The bundle references original weights and statistics; normalization is performed
once by the upstream processor. Inputs are two rotated uint8 RGB cameras (256px)
and an 8D end-effector state. Output is a 32-step chunk of native 7D delta actions.
Training through `vla.train` is not implemented for this adapter.

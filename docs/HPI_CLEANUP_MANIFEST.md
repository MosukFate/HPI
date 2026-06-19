

## Public API Cleanup

- Public segmentor registries are `HPI_CLIP` and `HPI_EVA_CLIP`.
- Public CLIP backbone registry is `HPIClipVisionTransformer`.
- EVA uses `model_revised_name='hpi'` and `EVAVisionTransformerHPI`.
- Legacy sequence-mixer optional kernels are removed from the active HPI import path.
- Old checkpoints should be converted with `tools/convert_hpi_checkpoint_names.py` before loading.

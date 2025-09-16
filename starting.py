
python -m pip install --upgrade pip

pip install mistralrs
pip install mistralrs-cuda -v
pip install huggingface_hub


export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export HF_MODULES_CACHE=/workspace/hf_cache/modules
export HF_ASSETS_CACHE=/workspace/hf_cache/assets

echo $HF_HOME
echo $HUGGINGFACE_HUB_CACHE
echo $TRANSFORMERS_CACHE
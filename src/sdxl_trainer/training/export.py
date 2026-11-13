"""Convert a completed local SDXL-base Diffusers pipeline into one safetensors file.

Architecture mapping follows the public SDXL checkpoint layout documented by
Hugging Face Diffusers' convert_diffusers_to_original_sdxl.py. This implementation
loads safetensors shards only, validates mappings and writes atomically. It never
loads pickle checkpoint files or contacts a model hub.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
import re
import sys

_UNET_DIRECT = {
    "time_embedding.linear_1": "time_embed.0", "time_embedding.linear_2": "time_embed.2",
    "conv_in": "input_blocks.0.0", "conv_norm_out": "out.0", "conv_out": "out.2",
    "add_embedding.linear_1": "label_emb.0.0", "add_embedding.linear_2": "label_emb.0.2",
}
_RESNET = {"norm1": "in_layers.0", "conv1": "in_layers.2", "norm2": "out_layers.0", "conv2": "out_layers.3", "time_emb_proj": "emb_layers.1", "conv_shortcut": "skip_connection"}


def _replace_first(value, mapping):
    first, _, rest = value.partition(".")
    return mapping.get(first, first) + ("." + rest if rest else "")


def unet_key(key):
    for source, target in _UNET_DIRECT.items():
        if key.startswith(source + "."):
            return "model.diffusion_model." + target + key[len(source):]
    match = re.fullmatch(r"(down_blocks|up_blocks)\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)", key)
    if match:
        direction, block, kind, layer, tail = match.groups()
        block, layer = int(block), int(layer)
        if direction == "down_blocks":
            prefix = f"input_blocks.{3 * block + layer + 1}"
        else:
            prefix = f"output_blocks.{3 * block + layer}"
        tail = _replace_first(tail, _RESNET) if kind == "resnets" else tail
        return f"model.diffusion_model.{prefix}.{0 if kind == 'resnets' else 1}.{tail}"
    match = re.fullmatch(r"down_blocks\.(\d+)\.downsamplers\.0\.conv\.(weight|bias)", key)
    if match:
        block, tail = match.groups()
        return f"model.diffusion_model.input_blocks.{3 * (int(block) + 1)}.0.op.{tail}"
    match = re.fullmatch(r"up_blocks\.(\d+)\.upsamplers\.0\.conv\.(weight|bias)", key)
    if match:
        block, tail = match.groups()
        return f"model.diffusion_model.output_blocks.{3 * int(block) + 2}.2.conv.{tail}"
    match = re.fullmatch(r"mid_block\.(resnets|attentions)\.(\d+)\.(.+)", key)
    if match:
        kind, layer, tail = match.groups()
        index = int(layer) * 2 if kind == "resnets" else 1
        tail = _replace_first(tail, _RESNET) if kind == "resnets" else tail
        return f"model.diffusion_model.middle_block.{index}.{tail}"
    raise ValueError(f"Unsupported SDXL UNet tensor: {key}")


def vae_key(key):
    key = key.replace("conv_shortcut", "nin_shortcut").replace("conv_norm_out", "norm_out")
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"encoder.down.{m[1]}.block.{m[2]}.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"decoder.up.{3-int(m[1])}.block.{m[2]}.", key)
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", lambda m: f"encoder.down.{m[1]}.downsample.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f"decoder.up.{3-int(m[1])}.upsample.", key)
    key = re.sub(r"mid_block\.resnets\.(\d+)\.", lambda m: f"mid.block_{int(m[1])+1}.", key)
    key = key.replace("mid_block.attentions.0.", "mid.attn_1.")
    if ".mid.attn_1." in key:
        key = key.replace("group_norm.", "norm.").replace("to_q.", "q.").replace("to_k.", "k.").replace("to_v.", "v.").replace("to_out.0.", "proj_out.")
    if "down_blocks" in key or "up_blocks" in key or "mid_block" in key or "attentions" in key:
        raise ValueError(f"Unsupported VAE tensor: {key}")
    return "first_stage_model." + key


def text_encoder_1_key(key):
    # Transformers releases can save CLIP-L either with or without the model
    # wrapper prefix. SDXL's single-file layout requires it in both cases.
    if not key.startswith("text_model."):
        key = "text_model." + key
    return "conditioner.embedders.0.transformer." + key


def text_encoder_2_key(key):
    key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
    key = key.replace("layer_norm1.", "ln_1.").replace("layer_norm2.", "ln_2.")
    key = key.replace("mlp.fc1.", "mlp.c_fc.").replace("mlp.fc2.", "mlp.c_proj.")
    key = key.replace("self_attn.", "attn.")
    key = key.replace("text_model.final_layer_norm.", "ln_final.")
    key = key.replace("text_model.embeddings.token_embedding.weight", "token_embedding.weight")
    key = key.replace("text_model.embeddings.position_embedding.weight", "positional_embedding")
    if key == "text_projection.weight":
        key = "text_projection"
    return "conditioner.embedders.1.model." + key


def _component_files(folder):
    candidates = [folder / "diffusion_pytorch_model.safetensors.index.json", folder / "model.safetensors.index.json"]
    for index in candidates:
        if index.is_file():
            data = json.loads(index.read_text())
            filenames = sorted(set(data["weight_map"].values()))
            if any(Path(name).name != name or not name.endswith(".safetensors") for name in filenames):
                raise ValueError("Unsafe or unsupported shard filename in model index.")
            files = [folder / name for name in filenames]
            if not all(file.is_file() for file in files):
                raise ValueError(f"Missing safetensors shard in {folder}.")
            return files
    for filename in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        candidate = folder / filename
        if candidate.is_file():
            return [candidate]
    raise ValueError(f"No standard safetensors weights found in {folder}. Pick a completed Diffusers output from this app.")


def export_checkpoint(model_path, output_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    model = Path(model_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.suffix != ".safetensors":
        raise ValueError("Checkpoint output must end in .safetensors.")
    if output.exists():
        raise ValueError("Checkpoint output already exists; choose a new filename.")
    if model == output or model in output.parents:
        raise ValueError("Choose an output filename outside the source model folder.")
    config = json.loads((model / "unet" / "config.json").read_text())
    if config.get("in_channels") != 4 or len(config.get("block_out_channels", [])) != 3 or config.get("cross_attention_dim") != 2048:
        raise ValueError("Only the standard SDXL base architecture is supported by this exporter.")
    result = {}
    qkv = {}
    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        print(json.dumps({"type": "status", "message": f"Mapping {component} tensors."}), flush=True)
        for shard in _component_files(model / component):
            with safe_open(str(shard), framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    tensor = stream.get_tensor(key)
                    if component == "unet":
                        converted = unet_key(key)
                    elif component == "vae":
                        converted = vae_key(key)
                        if re.search(r"\.mid\.attn_1\.(q|k|v|proj_out)\.weight$", converted) and tensor.ndim == 2:
                            tensor = tensor[:, :, None, None]
                    elif component == "text_encoder":
                        converted = text_encoder_1_key(key)
                    else:
                        match = re.fullmatch(r"(.+\.self_attn)\.([qkv])_proj\.(weight|bias)", key)
                        if match:
                            prefix, letter, kind = match.groups()
                            qkv.setdefault((prefix, kind), {})[letter] = tensor
                            continue
                        converted = text_encoder_2_key(key)
                        if key == "text_projection.weight":
                            tensor = tensor.T.contiguous()
                    if converted in result:
                        raise ValueError(f"Duplicate mapped tensor: {converted}")
                    result[converted] = tensor.contiguous()
    for (prefix, kind), tensors in qkv.items():
        if set(tensors) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete CLIP-G QKV group: {prefix}.{kind}")
        converted = text_encoder_2_key(prefix + ".in_proj_" + kind)
        result[converted] = torch.cat([tensors[x] for x in "qkv"], dim=0).contiguous()
    if not qkv:
        raise ValueError("CLIP-G contained no attention projections; source model is incompatible.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(result, str(temporary), metadata={"format": "pt", "sdxl_trainer": "1", "architecture": "stable-diffusion-xl-base"})
        # Same-filesystem link is atomic and never replaces an existing filename.
        # Each exporter owns a unique temporary file, including concurrent runs.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"type": "complete", "path": str(output), "message": "Single-file SDXL checkpoint exported."}), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--output_path", "--output-path", required=True)
    args = parser.parse_args(argv)
    try:
        export_checkpoint(args.model_path, args.output_path)
        return 0
    except Exception as exc:
        print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

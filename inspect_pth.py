#!/usr/bin/env python3
"""
inspect_pth.py
Small utility to inspect a PyTorch .pth file and print a summary.

Usage: python inspect_pth.py

This script will try to import torch. If torch is unavailable it will
print basic file metadata and a hexdump of the first bytes.
"""
import os
import sys
from pathlib import Path
import argparse
import numpy as np

PTH = Path(__file__).with_name('matt_pcnn.pth')


def human_shape(tensor):
    try:
        return tuple(tensor.size())
    except Exception:
        return None


def main():
    if not PTH.exists():
        print(f"File not found: {PTH}")
        sys.exit(2)

    try:
        import torch
    except Exception as e:
        print("torch not available (or failed to import):", e)
        print("Falling back to raw file metadata and header hexdump.")
        st = PTH.stat()
        print(f"Path: {PTH.resolve()}")
        print(f"Size: {st.st_size} bytes")
        with PTH.open('rb') as f:
            head = f.read(256)
        print("First 256 bytes (hex):\n", head.hex())
        return

    # Some checkpoints were saved as full objects requiring the original
    # class to be allowed for safe loading. Try to allowlist PCNN_3Branch
    # from the local Model module before loading. Also set weights_only=False
    # so full-object loads are attempted (this can execute code; only do for
    # trusted files).
    # Some checkpoints reference classes saved under the `__main__` module
    # (for example '__main__.PCNN_3Branch'). PyTorch's safe loading blocks
    # these by default. Easiest safe workaround for a local, trusted file is
    # to import the class and inject it into the __main__ module so the name
    # resolves during unpickling.
    try:
        import Model
        PCNN = getattr(Model, 'PCNN_3Branch', None)
        if PCNN is not None:
            import types
            import importlib
            main_mod = sys.modules.get('__main__')
            if main_mod is None:
                main_mod = types.ModuleType('__main__')
                sys.modules['__main__'] = main_mod
            # inject class into __main__ namespace
            setattr(main_mod, 'PCNN_3Branch', PCNN)
    except Exception:
        # Ignore injection failures; we'll still try to load normally
        pass

    try:
        # Attempt full-object load (weights_only=False) when available.
        load_kwargs = dict(map_location='cpu')
        # some torch versions support weights_only arg
        try:
            obj = torch.load(str(PTH), **{**load_kwargs, **{'weights_only': False}})
        except TypeError:
            # weights_only not supported by this torch; fall back
            obj = torch.load(str(PTH), **load_kwargs)
    except Exception as e:
        print("torch.load failed:", e)
        sys.exit(3)

    # CLI: allow the user to request more detailed outputs or to save artifacts
    parser = argparse.ArgumentParser(description='Inspect a PyTorch .pth file (model or state_dict)')
    parser.add_argument('--list', action='store_true', help='List parameter names, shapes, dtypes and numel')
    parser.add_argument('--stats', action='store_true', help='Compute min/max/mean/std for tensor parameters')
    parser.add_argument('--save-state', metavar='PATH', help='Save state_dict to PATH (torch.save)')
    parser.add_argument('--save-npz', metavar='PATH', help='Save all tensors to a .npz file at PATH')
    parser.add_argument('--save-json', metavar='PATH', help='Save a detailed JSON summary to PATH')
    parser.add_argument('--plot-after', action='store_true', help='If JSON is saved, run plot_from_json.py to generate plots')
    parser.add_argument('--topk', type=int, default=20, help='When listing, print only the first TOPK params (0 = all)')
    args = parser.parse_args()

    # Normalize to a state_dict-like mapping: if obj is a model, use state_dict()
    state = None
    if isinstance(obj, dict):
        state = obj
    else:
        # Try common model API
        try:
            state = obj.state_dict()
        except Exception:
            # Fallback: if it's an object with attributes, try __dict__
            try:
                state = {k: v for k, v in vars(obj).items()}
            except Exception:
                state = { 'object': obj }

    # Helper to compute tensor stats safely
    def tensor_stats(t):
        try:
            arr = t.detach().cpu().numpy()
            return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())
        except Exception:
            return None

    # Print a brief header
    print(f"Loaded object type: {type(obj)}")
    print(f"Parameters / entries: {len(state)}")

    # Listing and stats
    names = list(state.keys())
    if args.topk and args.topk > 0:
        display_names = names[:args.topk]
    else:
        display_names = names

    total_params = 0
    for name in display_names:
        val = state[name]
        info = ''
        try:
            if hasattr(val, 'size'):
                shape = tuple(val.size())
                numel = int(val.numel())
                total_params += numel
                dtype = getattr(val, 'dtype', None)
                info = f'shape={shape} dtype={dtype} numel={numel}'
                if args.stats:
                    stats = tensor_stats(val)
                    if stats is not None:
                        info += f' min={stats[0]:.6g} max={stats[1]:.6g} mean={stats[2]:.6g} std={stats[3]:.6g}'
            else:
                info = f'type={type(val)}'
        except Exception as e:
            info = f'error inspecting value: {e}'
        print(f"- {name}: {info}")

    # If not listing all, show summary for remaining count
    if args.topk and args.topk > 0 and len(names) > args.topk:
        remaining = len(names) - args.topk
        print(f"... and {remaining} more parameters (use --topk 0 to show all)")

    print(f"Total params (counted in displayed set): {total_params}")

    # Save state_dict if requested
    if args.save_state:
        try:
            # If state is a model object, make sure we save a dict
            torch.save(state, args.save_state)
            print(f"Saved state-like object to {args.save_state}")
        except Exception as e:
            print(f"Failed to save state to {args.save_state}: {e}")

    if args.save_npz:
        try:
            npdict = {}
            for k, v in state.items():
                try:
                    if hasattr(v, 'cpu'):
                        npdict[k] = v.detach().cpu().numpy()
                    else:
                        # Skip non-tensor objects
                        continue
                except Exception:
                    continue
            np.savez_compressed(args.save_npz, **npdict)
            print(f"Saved tensors to {args.save_npz} (.npz)")
        except Exception as e:
            print(f"Failed to save .npz: {e}")

    if args.save_json:
        try:
            import json
            js = []
            for k, v in state.items():
                entry = {'name': k}
                try:
                    if hasattr(v, 'size'):
                        shape = tuple(v.size())
                        numel = int(v.numel())
                        entry.update({'shape': shape, 'numel': numel, 'dtype': str(getattr(v, 'dtype', None))})
                        if args.stats:
                            stats = tensor_stats(v)
                            if stats is not None:
                                entry.update({'min': stats[0], 'max': stats[1], 'mean': stats[2], 'std': stats[3]})
                    else:
                        entry.update({'type': str(type(v))})
                except Exception as e:
                    entry.update({'error': str(e)})
                js.append(entry)
            summary = {'object_type': str(type(obj)), 'entries': js, 'total_entries': len(js)}
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved JSON summary to {args.save_json}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")

        if args.plot_after:
            # try to run plot_from_json.py with the saved JSON path
            try:
                import subprocess
                script = Path(__file__).with_name('plot_from_json.py')
                if script.exists():
                    subprocess.check_call([sys.executable, str(script), args.save_json])
                else:
                    print('plot_from_json.py not found; skipping plotting')
            except Exception as e:
                print('Failed to run plot_from_json.py:', e)

    # End main


if __name__ == '__main__':
    main()

"""Execute the notebook's code cells headlessly, to regenerate results/ plots.

Lives in the repo rather than a scratch directory so it survives session
restarts. Skips cell 0 (the Colab clone/install) and every markdown cell.

Everything runs under a `__main__` guard. Without it, merely *importing* this
module executed the whole notebook — a full spatial evaluation over the real
dataset — so any dead-code scanner, IDE indexer or `pytest --collect-only`
that touched the file kicked off a training-scale run.
"""
import json, sys, matplotlib


def main():
    matplotlib.use("Agg")

    nb = json.load(open("Signature_Verification_Pipeline.ipynb"))
    ns = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code" or idx == 0:
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        print(f"\n===== CELL {idx} ({cell['id']}) =====", flush=True)
        try:
            exec(  # skylos: ignore[SKY-D202] running the notebook is this tool's purpose
                compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception:
            import traceback; traceback.print_exc()
            sys.exit(f"CELL {idx} FAILED")
    print("\nALL CELLS OK")


if __name__ == "__main__":
    main()

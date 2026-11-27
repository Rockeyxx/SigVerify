"""Execute the notebook's code cells headlessly to regenerate `results/` plots (skips cell 0)."""
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

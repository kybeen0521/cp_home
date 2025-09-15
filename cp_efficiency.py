import logging
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RHO: float = 1000.0  # Water density [kg/m³]
G: float = 9.81      # Gravity [m/s²]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace("\n", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    return df


def plot_efficiency(df: pd.DataFrame) -> None:
    plt.plot(
        df["q_m3s"],
        df["efficiency"],
        "-o",
        color="green",
        markersize=6,
        linewidth=2,
        markerfacecolor="lime",
        label="Efficiency",
    )

    max_idx = df["efficiency"].idxmax()
    max_q = df.loc[max_idx, "q_m3s"]
    max_eff = df.loc[max_idx, "efficiency"]

    plt.scatter(max_q, max_eff, color="red", s=100, zorder=5, label="Max Efficiency")
    texts = [plt.text(max_q, max_eff, f"({max_q:.4f}, {max_eff:.2f})",
                      fontsize=10, color="red")]

    adjust_text(
        texts,
        only_move={"points": "y", "text": "y"},
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
    )

    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Efficiency [%]")
    plt.title("Pump Efficiency Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info(f"Maximum efficiency: {max_eff:.2f}% at Q = {max_q:.4f} m³/s")


# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def main() -> None:
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[("Excel files", "*.xlsx *.xls")],
    )

    if not file_path:
        logging.error("No file selected. Exiting.")
        return

    data_file = Path(file_path)
    logging.info(f"File selected: {data_file}")

    try:
        df = pd.read_excel(data_file)
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        return

    df = clean_columns(df)

    try:
        torque_col = next(c for c in df.columns if "Torque" in c)
        rpm_col = next(c for c in df.columns if "Speed" in c or "RPM" in c)
        flow_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
        p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
        p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
        v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
        v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)
    except StopIteration:
        logging.error(f"Required columns not found. Available columns:\n{list(df.columns)}")
        return

    df["q_m3s"] = df[flow_col] / 1000.0
    df["ha"] = (
        (df[p2_col] - df[p1_col]) / (RHO * G)
        + 2
        + (df[v2_col] ** 2 - df[v1_col] ** 2) / (2 * G)
    )
    df["w_hydraulic"] = RHO * G * df["q_m3s"] * df["ha"]
    df["omega"] = (df[rpm_col] * np.pi) / 30.0
    df["w_shaft"] = df[torque_col] * df["omega"]
    df["efficiency"] = (df["w_hydraulic"] / df["w_shaft"]) * 100.0

    df_sorted = df.sort_values("q_m3s")

    logging.info(
        "\n=== Efficiency Results ===\n"
        + df_sorted[["q_m3s", "efficiency"]].to_string(
            index=False, header=["Flow rate [m³/s]", "Efficiency [%]"]
        )
    )

    plot_efficiency(df_sorted)


if __name__ == "__main__":
    main()

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
G: float = 9.81  # Gravity [m/s²]

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


def find_stable_bep(df: pd.DataFrame) -> pd.Series:
    """
    안정적인 BEP 찾기:
    최대 효율 근처에서 앞뒤 지점과 y값 차이가 최소인 지점을 선택
    """
    max_eff = df["efficiency"].max()
    candidates = df[df["efficiency"] >= 0.99 * max_eff].copy()
    candidates["y_diff"] = candidates["efficiency"].diff().abs().fillna(0) + \
                           candidates["efficiency"].diff(-1).abs().fillna(0)
    stable_bep = candidates.loc[candidates["y_diff"].idxmin()]
    return stable_bep


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

    # 안정적인 BEP 표시
    bep_row = find_stable_bep(df)
    bep_q = bep_row["q_m3s"]
    bep_eff = bep_row["efficiency"]

    plt.scatter(bep_q, bep_eff, color="red", s=100, zorder=5)
    texts = [plt.text(
        bep_q, bep_eff + 0.5, f"BEP\n(Q={bep_q:.4f}, Eff={bep_eff:.2f}%)",
        fontsize=10, color="red", fontweight="bold", ha="center"
    )]

    adjust_text(
        texts,
        only_move={"points": "y", "text": "y"},
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        expand_points=(1.2, 1.2),
    )

    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Efficiency [%]")
    plt.title("Pump Efficiency Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info(f"Stable BEP: {bep_eff:.2f}% at Q = {bep_q:.4f} m³/s")


# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def select_file() -> Path:
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel or CSV file",
        filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")],
    )
    return Path(file_path) if file_path else None


def main() -> None:
    data_file = select_file()
    if not data_file:
        logging.error("No file selected. Exiting.")
        return

    logging.info(f"File selected: {data_file}")

    # ----------------------------
    # Read file (Excel or CSV)
    # ----------------------------
    try:
        ext = data_file.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(data_file)
        elif ext == ".csv":
            try:
                df = pd.read_csv(data_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(data_file, encoding="cp949")
        else:
            logging.error(f"Unsupported file type: {ext}")
            return
    except Exception as e:
        logging.error(f"Failed to read file: {e}")
        return

    df = clean_columns(df)

    # ----------------------------
    # Column detection
    # ----------------------------
    try:
        torque_col = next(c for c in df.columns if "Torque" in c)
        rpm_col = next(c for c in df.columns if "Speed" in c or "RPM" in c)
        flow_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
        p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
        p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
        v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
        v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)
        he_col = next(c for c in df.columns if "Elevation Head" in c)
    except StopIteration:
        logging.error(f"Required columns not found. Available columns:\n{list(df.columns)}")
        return

    # ----------------------------
    # Calculations (BEP)
    # ----------------------------
    df["p_in_Pa"] = df[p1_col] * 1000.0  # kPa → Pa
    df["p_out_Pa"] = df[p2_col] * 1000.0

    df["q_m3s"] = df[flow_col] / 1000.0  # L/s → m³/s

    df["ha"] = ((df["p_out_Pa"] - df["p_in_Pa"]) / (RHO * G)
                + df[he_col]
                + (df[v2_col] ** 2 - df[v1_col] ** 2) / (2 * G))

    df["w_hydraulic"] = RHO * G * df["q_m3s"] * df["ha"]
    df["omega"] = df[rpm_col] * np.pi / 30.0
    df["w_shaft"] = df[torque_col] * df["omega"]

    df["efficiency"] = (df["w_hydraulic"] / df["w_shaft"]) * 100.0
    df["efficiency"] = df["efficiency"].clip(lower=0, upper=100)  # 0~100% 제한

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

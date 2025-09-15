import logging
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
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


def calc_shaft_power(torque: pd.Series, rpm: pd.Series) -> pd.Series:
    omega = rpm * np.pi / 30
    return torque * omega


def plot_shaft_curve(df: pd.DataFrame, flow_col: str, w_col: str) -> None:
    plt.plot(
        df[flow_col],
        df[w_col],
        "-o",
        color="royalblue",
        markersize=6,
        linewidth=2,
        markerfacecolor="orange",
    )
    plt.xlabel("Flow rate Q [l/s]", fontsize=12)
    plt.ylabel("Shaft Power Wshaft [W]", fontsize=12)
    plt.title("Shaft Power Curve", fontsize=14)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def main() -> None:
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="엑셀 데이터 파일 선택",
        filetypes=[("Excel files", "*.xlsx *.xls")],
    )
    root.destroy()

    if not file_path:
        logging.error("파일을 선택하지 않았습니다.")
        return

    data_file = Path(file_path)
    logging.info(f"선택된 파일: {data_file}")

    try:
        df = pd.read_excel(data_file)
    except Exception as e:
        logging.error(f"엑셀 파일 읽기 실패: {e}")
        return

    df = clean_columns(df)

    torque_col = next(
        (c for c in df.columns if "Torque" in c or "t [Nm]" in c or "Motor Torque" in c),
        None,
    )
    rpm_col = next(
        (c for c in df.columns if "Pump Speed" in c or "n [rpm]" in c),
        None,
    )
    flow_col = next(
        (c for c in df.columns if "Flow" in c or "Q" in c),
        None,
    )

    if not all([torque_col, rpm_col, flow_col]):
        logging.error("Torque, RPM 또는 Flow 컬럼을 찾을 수 없습니다.")
        logging.error(f"존재하는 컬럼들: {list(df.columns)}")
        return

    df["Wshaft"] = calc_shaft_power(df[torque_col], df[rpm_col])
    df_sorted = df.sort_values(by=flow_col)

    logging.info(
        "\n=== Shaft Power Results ===\n"
        + df_sorted[[flow_col, "Wshaft"]].to_string(
            index=False, header=["Flow rate [l/s]", "Shaft Power [W]"]
        )
    )

    plot_shaft_curve(df_sorted, flow_col, "Wshaft")


if __name__ == "__main__":
    main()

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

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

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace, newline, tab, and special characters from DataFrame column names."""
    df.columns = (
        df.columns
        .str.replace("\n", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    return df


def calc_shaft_power(torque: pd.Series, rpm: pd.Series) -> pd.Series:
    """
    Calculate shaft power Wshaft [W] from torque and RPM.

    Args:
        torque (pd.Series): Torque [Nm]
        rpm (pd.Series): Pump speed [RPM]

    Returns:
        pd.Series: Shaft power [W]
    """
    omega: pd.Series = rpm * np.pi / 30  # rad/s
    Wshaft: pd.Series = torque * omega
    return Wshaft


def plot_shaft_curve(df: pd.DataFrame, flow_col: str, W_col: str) -> None:
    """
    Plot shaft power curve.

    Args:
        df (pd.DataFrame): DataFrame containing flow rate and shaft power columns
        flow_col (str): Column name for flow rate
        W_col (str): Column name for shaft power
    """
    plt.plot(df[flow_col], df[W_col], '-o', color='royalblue', markersize=6,
             linewidth=2, markerfacecolor='orange')
    plt.xlabel("Flow rate Q [l/s]", fontsize=12)
    plt.ylabel("Shaft Power Wshaft [W]", fontsize=12)
    plt.title("Shaft Power Curve", fontsize=14)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def main() -> None:
    """Main function to calculate and plot shaft power curve."""

    # 파일 선택 창 열기
    root = Tk()
    root.withdraw()  # Tkinter 기본 창 숨기기
    file_path = filedialog.askopenfilename(
        title="엑셀 데이터 파일 선택",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    root.destroy()

    if not file_path:
        logging.error("❌ 파일을 선택하지 않았습니다.")
        return

    data_file = Path(file_path)
    logging.info(f"✅ 선택된 파일: {data_file}")

    df: pd.DataFrame = pd.read_excel(data_file)
    df = clean_columns(df)

    # 컬럼 자동 탐색
    torque_col = next((c for c in df.columns if "Torque" in c or "t [Nm]" in c or "Motor Torque" in c), None)
    rpm_col = next((c for c in df.columns if "Pump Speed" in c or "n [rpm]" in c), None)
    flow_col = next((c for c in df.columns if "Flow" in c or "Q" in c), None)

    if not all([torque_col, rpm_col, flow_col]):
        logging.error("Torque, RPM 또는 Flow 컬럼을 찾을 수 없습니다. 파일 컬럼명을 확인하세요:")
        logging.error(list(df.columns))
        return

    # Shaft Power 계산
    df["Wshaft"] = calc_shaft_power(df[torque_col], df[rpm_col])

    # Flow 기준 정렬
    df_sorted: pd.DataFrame = df.sort_values(by=flow_col)

    # 결과 출력
    logging.info("\n=== Shaft Power Results ===\n" +
                 df_sorted[[flow_col, "Wshaft"]].to_string(index=False,
                                                           header=["Flow rate [l/s]", "Shaft Power [W]"]))

    # 그래프 출력
    plot_shaft_curve(df_sorted, flow_col, "Wshaft")


if __name__ == "__main__":
    main()

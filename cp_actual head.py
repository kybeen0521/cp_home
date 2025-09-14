import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tkinter import Tk, filedialog


# ----------------------------
# Configuration
# ----------------------------
Z_DIFF: float = 2.0             # Height difference [m]
G: float = 9.81                 # Gravity [m/s²]
RHO: float = 1000               # Water density [kg/m³]
RAMDA: float = RHO * G / 1000   # Pressure conversion factor [kPa]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------
# Utility Functions
# ----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing whitespace, newline, and special characters."""
    df.columns = (
        df.columns
        .str.replace("\n", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    return df


def calc_ha(p1: pd.Series, p2: pd.Series,
            v1: pd.Series, v2: pd.Series,
            z_diff: float = Z_DIFF,
            ramda: float = RAMDA,
            g: float = G) -> pd.Series:
    """Calculate actual head [m]."""
    velocity_term = (v2**2 - v1**2) / (2 * g)
    return (p2 - p1) / ramda + z_diff + velocity_term


def ordinal(n: int) -> str:
    """Convert integer to ordinal string (1 -> 1st, 2 -> 2nd, etc.)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def plot_actual_head_with_highlight(df: pd.DataFrame) -> None:
    """Plot Actual Head Curve with highlighted increases and print details."""
    df['delta_ha'] = df['ha'].diff()  # 이전 지점 대비 상승량
    highlight_idx = df.index[df['delta_ha'] > 0.01]  # 상승 임계값 0.01 m

    plt.plot(df["Q_m3s"], df["ha"], "-o",
             color="royalblue", markersize=6, linewidth=2,
             markerfacecolor="orange", label="Actual Head")

    plt.scatter(df.loc[highlight_idx, "Q_m3s"], df.loc[highlight_idx, "ha"],
                color="red", s=50, label="Increase")

    texts = []
    for i, idx in enumerate(highlight_idx, start=1):
        x = df.loc[idx, "Q_m3s"]
        y = df.loc[idx, "ha"]
        increase = df.loc[idx, 'delta_ha']
        t = plt.text(x, y, f"{ordinal(i)} +{increase:.2f} m", fontsize=8, ha='center')
        texts.append(t)

        # 상승 지점 정보 출력
        logging.info(f"{ordinal(i)} 지점: +{increase:.2f} m 상승 (Q={x:.5f} m³/s)")

    adjust_text(texts, only_move={'points':'y', 'text':'y'},
                arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Actual head ha [m]")
    plt.title("Actual Head Curve with Highlighted Increase")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info(f"총 상승 횟수: {len(highlight_idx)}회")


# ----------------------------
# Main Workflow
# ----------------------------
def main() -> None:
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

    df = pd.read_excel(data_file)
    df = clean_columns(df)

    # 자동 컬럼 탐색
    q_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
    p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
    p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
    v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
    v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)

    df["Q_m3s"] = df[q_col] / 1000
    df["ha"] = calc_ha(df[p1_col], df[p2_col], df[v1_col], df[v2_col])
    df_sorted = df.sort_values(by="Q_m3s")

    logging.info("\n=== Actual Head Results ===\n" +
                 df_sorted[["Q_m3s", "ha"]].to_string(index=False,
                 header=["Flow rate [m³/s]", "Actual head [m]"]))

    plot_actual_head_with_highlight(df_sorted)


if __name__ == "__main__":
    main()

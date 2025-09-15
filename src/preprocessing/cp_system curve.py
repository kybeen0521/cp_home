import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tkinter import Tk, filedialog


# Configuration Classes
class PipeConfig:
    """Pipe parameters and constants."""
    D: float = 0.02          # Pipe diameter [m]
    L: float = 2.0           # Pipe length [m]
    EPS: float = 0.00015     # Absolute roughness [m]
    A: float = np.pi * (D / 2) ** 2  # Cross-sectional area [m²]
    KL_SUM: float = 1.8      # Sum of minor loss coefficients
    Z_DIFF: float = 2.0      # Elevation difference [m]
    G: float = 9.81          # Gravity [m/s²]


class FluidProperties:
    """Fluid properties (water at 20°C)."""
    RHO: float = 998.2       # Density [kg/m³]
    MU: float = 0.001003     # Dynamic viscosity [Pa·s]


class PlotConfig:
    """Plotting configuration."""
    FIGSIZE = (10, 6)
    FONT_FAMILY = "Times New Roman"
    MARKER_SIZE = 6
    LINE_WIDTH = 2
    THEME = "darkgrid"


# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Utility Functions
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace and unwanted characters from DataFrame column names."""
    df.columns = (
        df.columns.str.replace("\n", "", regex=False)
                  .str.replace("\t", "", regex=False)
                  .str.replace("\xa0", "", regex=False)
                  .str.strip()
    )
    return df


def velocity(Q: float, config: PipeConfig) -> float:
    """Convert volumetric flow rate [m³/s] to average velocity [m/s]."""
    return 4 * Q / (np.pi * config.D ** 2)


def reynolds(rho: float, mu: float, Q: float, config: PipeConfig) -> float:
    """Calculate Reynolds number."""
    V = velocity(Q, config)
    return rho * V * config.D / mu


def friction_factor(Re: float, config: PipeConfig) -> float:
    """Calculate Darcy friction factor using laminar or Haaland formula."""
    if Re < 2000:
        return 64.0 / Re
    return (-1.8 * np.log10((config.EPS / (3.7 * config.D)) + (5.74 / (Re ** 0.9)))) ** -2


def calc_system_head(Q: pd.Series, fluid: FluidProperties, config: PipeConfig) -> pd.Series:
    """
    Calculate system curve head.

    H_system = Z_DIFF + [(f*L/D + ΣKL) / (2 * g * A²)] * Q²
    """
    H_sys = []
    for q in Q:
        Re = reynolds(fluid.RHO, fluid.MU, q, config)
        f = friction_factor(Re, config)
        k = (f * config.L / config.D + config.KL_SUM) / (2 * config.G * config.A ** 2)
        H_sys.append(config.Z_DIFF + k * q ** 2)
    return pd.Series(H_sys, index=Q.index)


def plot_system_curve(Q: pd.Series, H_sys: pd.Series) -> None:
    """Plot the system curve."""
    sns.set_theme(style=PlotConfig.THEME)
    plt.figure(figsize=PlotConfig.FIGSIZE)
    plt.plot(Q, H_sys, "-o", color="purple", label="System Curve",
             markersize=PlotConfig.MARKER_SIZE, linewidth=PlotConfig.LINE_WIDTH)
    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Head [m]")
    plt.title("System Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_flow_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically find the flow column in the DataFrame."""
    return next((c for c in df.columns if "Flow" in c and "Q" in c), None)


# Main Workflow
def main() -> None:
    config = PipeConfig()
    fluid = FluidProperties()

    try:
        # Tkinter File Selection
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not file_path:
            logging.error("No file selected. Exiting...")
            return

        df = pd.read_excel(file_path)
        df = clean_columns(df)

        flow_col = find_flow_column(df)
        if flow_col is None:
            logging.error("Flow column not found in Excel file!")
            return

        # Convert L/s → m³/s
        df["Q_m3s"] = df[flow_col] / 1000

        # Calculate system curve
        df["H_system"] = calc_system_head(df["Q_m3s"], fluid, config)

        logging.info("\n=== System Curve Results ===\n" +
                     df[["Q_m3s", "H_system"]].to_string(
                         index=False,
                         header=["Flow rate [m³/s]", "System Head [m]"]
                     ))

        # Plot
        plot_system_curve(df["Q_m3s"], df["H_system"])

    except FileNotFoundError as fnf_err:
        logging.error(f"FileNotFoundError: {fnf_err}")
    except pd.errors.ExcelFileError as xl_err:
        logging.error(f"ExcelFileError: {xl_err}")
    except Exception as e:
        logging.exception(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

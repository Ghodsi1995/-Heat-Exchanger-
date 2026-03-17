from __future__ import annotations

"""
Heat_Exchanger Streamlit app - version 2

Why this version exists
-----------------------
The previous version used st.dataframe(), which makes Streamlit import pyarrow.
On the user's Windows environment, pyarrow failed to load one of its DLLs.
That means the UI table widget crashed before the thermodynamic model could finish.

This version avoids pandas and st.dataframe completely.
It renders results using Streamlit text, metrics, columns, and simple HTML tables.

Thermodynamic model
-------------------
- Stream 1 = Water (hot side)
- Stream 2 = CH4   (cold side)
- Input 1  = water inlet   = left nozzle on the image
- Output 1 = water outlet  = right nozzle on the image
- Input 2  = CH4 inlet     = bottom nozzle on the image
- Output 2 = CH4 outlet    = top nozzle on the image

Cantera usage
-------------
- CH4 is loaded from a YAML phase created by this script.
- H2O is also included in the YAML for reference consistency.
- The actual water-side solve uses ct.Water(backend="Reynolds") because the water is at 140 bar.
  That is more physically appropriate than treating water as ideal-gas H2O.
  
  To run the app
  conda activate cantera
  streamlit run heat_exchanger_streamlit_app_v2_no_pyarrow.py    
"""

import html
import math
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import cantera as ct
    HAS_CANTERA = True
    CANTERA_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    HAS_CANTERA = False
    CANTERA_IMPORT_ERROR = str(exc)


APP_DIR = Path(__file__).resolve().parent
IMAGE_PATH = APP_DIR / "Heat_Exchanger.png"
YAML_PATH = APP_DIR / "heat_exchanger_ch4_h2o.yaml"


EMBEDDED_YAML = r"""
generator: YamlWriter
cantera-version: 3.2.0
description: |-
  Minimal CH4 / H2O thermo subset for a heat-exchanger app.
  CH4 and H2O species data are extracted from Cantera's gri30.yaml.
  The YAML is used here for methane ideal-gas properties and a reference H2O species.
  The actual water-side solve is done with ct.Water(backend="Reynolds").
phases:
  - name: methane_ig
    thermo: ideal-gas
    kinetics: gas
    reactions: none
    elements: [C, H]
    species: [CH4]
    state:
      T: 298.15
      P: 1 atm
      X: {CH4: 1.0}

  - name: steam_ig
    thermo: ideal-gas
    kinetics: gas
    reactions: none
    elements: [H, O]
    species: [H2O]
    state:
      T: 298.15
      P: 1 atm
      X: {H2O: 1.0}

species:
  - name: CH4
    composition: {C: 1, H: 4}
    thermo:
      model: NASA7
      temperature-ranges: [200.0, 1000.0, 3500.0]
      data:
        - [5.14987613, -0.0136709788, 4.91800599e-05, -4.84743026e-08,
           1.66693956e-11, -1.02466476e+04, -4.64130376]
        - [0.074851495, 0.0133909467, -5.73285809e-06, 1.22292535e-09,
           -1.0181523e-13, -9468.34459, 18.437318]
      note: L8/88
    transport:
      model: gas
      geometry: nonlinear
      well-depth: 141.4
      diameter: 3.746
      polarizability: 2.6
      rotational-relaxation: 13.0

  - name: H2O
    composition: {H: 2, O: 1}
    thermo:
      model: NASA7
      temperature-ranges: [200.0, 1000.0, 3500.0]
      data:
        - [4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09,
           1.77197817e-12, -3.02937267e+04, -0.849032208]
        - [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11,
           1.68200992e-14, -3.00042971e+04, 4.9667701]
      note: L8/89
    transport:
      model: gas
      geometry: nonlinear
      well-depth: 572.4
      diameter: 2.605
      dipole: 1.844
      rotational-relaxation: 4.0
"""


def ensure_yaml_exists() -> Path:
    """Write the embedded YAML to disk if it is not already present."""
    if not YAML_PATH.exists():
        YAML_PATH.write_text(EMBEDDED_YAML, encoding="utf-8")
    return YAML_PATH


# -----------------------------------------------------------------------------
# Cantera property models
# -----------------------------------------------------------------------------
def methane_phase(yaml_path: Path):
    """Load the methane ideal-gas phase from the generated YAML file."""
    return ct.Solution(str(yaml_path), "methane_ig")


def steam_phase(yaml_path: Path):
    """Load the H2O ideal-gas reference phase from the generated YAML file."""
    return ct.Solution(str(yaml_path), "steam_ig")


def water_phase():
    """Create the real-fluid water phase used for the high-pressure water side."""
    return ct.Water(backend="Reynolds")


def ch4_h_mass_from_yaml(yaml_path: Path, T_C: float, P_bar: float) -> float:
    """
    CH4 enthalpy [J/kg] from the YAML phase.

    Cantera steps:
    1) load methane phase from YAML
    2) set T, P, composition
    3) ask Cantera for enthalpy_mass
    """
    gas = methane_phase(yaml_path)
    gas.TPX = T_C + 273.15, P_bar * 1e5, {"CH4": 1.0}
    return float(gas.enthalpy_mass)


def ch4_T_from_hP_yaml(
    yaml_path: Path,
    h_target: float,
    P_bar: float,
    T_low_C: float = -100.0,
    T_high_C: float = 1400.0,
) -> float:
    """
    Recover CH4 temperature from known enthalpy and pressure using bisection.

    Why this function exists:
    - For the UA mode, CH4 outlet enthalpy is known from the energy balance.
    - The corresponding outlet temperature is not directly specified.
    - So we call Cantera repeatedly until the enthalpy at a trial temperature matches h_target.
    """

    def residual(T_C: float) -> float:
        return ch4_h_mass_from_yaml(yaml_path, T_C, P_bar) - h_target

    a = T_low_C
    b = T_high_C
    fa = residual(a)
    fb = residual(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(
            "CH4 temperature inversion failed. Increase the temperature bracket "
            "or check the specified enthalpy target."
        )

    for _ in range(200):
        m = 0.5 * (a + b)
        fm = residual(m)
        if abs(fm) < 1e-7:
            return m
        if fa * fm < 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


def water_h_mass_real(T_C: float, P_bar: float) -> float:
    """
    Real-fluid water enthalpy [J/kg] from Cantera at given temperature and pressure.
    """
    w = water_phase()
    w.TP = T_C + 273.15, P_bar * 1e5
    return float(w.enthalpy_mass)


def water_T_from_hP_real(h_mass: float, P_bar: float) -> float:
    """
    Recover water temperature [°C] from enthalpy and pressure using Cantera's HP setter.
    """
    w = water_phase()
    w.HP = h_mass, P_bar * 1e5
    return float(w.T - 273.15)


def h2o_h_mass_yaml_reference(yaml_path: Path, T_C: float, P_bar: float) -> float:
    """
    Ideal-gas H2O enthalpy [J/kg] from the YAML file.
    This is shown only as a reference value, not as the main water-side model.
    """
    steam = steam_phase(yaml_path)
    steam.TPX = T_C + 273.15, P_bar * 1e5, {"H2O": 1.0}
    return float(steam.enthalpy_mass)


# -----------------------------------------------------------------------------
# Heat-exchanger equations
# -----------------------------------------------------------------------------
def lmtd_countercurrent(dT1: float, dT2: float) -> float:
    """Counter-current log-mean temperature difference."""
    if dT1 <= 0.0 or dT2 <= 0.0:
        raise ValueError("Temperature cross occurred; counter-current LMTD is not defined.")
    if abs(dT1 - dT2) < 1e-12:
        return 0.5 * (dT1 + dT2)
    return (dT1 - dT2) / math.log(dT1 / dT2)


def solve_mode_given_output2(
    yaml_path: Path,
    *,
    mdot_1: float,
    T1_in_C: float,
    P1_in_bar: float,
    mdot_2: float,
    T2_in_C: float,
    P2_in_bar: float,
    T2_out_C: float,
    dP1_bar: float,
    dP2_bar: float,
) -> Dict[str, float | str]:
    """
    Mode A: Output 2 temperature is specified by the user.

    Method:
    1) Compute outlet pressures from pressure-drop inputs.
    2) Use Cantera to get CH4 inlet and outlet enthalpies.
    3) Compute duty from the CH4 side.
    4) Use energy balance to get water outlet enthalpy.
    5) Use Cantera real-water model to get water outlet temperature.
    """
    P1_out_bar = P1_in_bar - dP1_bar
    P2_out_bar = P2_in_bar - dP2_bar
    if P1_out_bar <= 0.0 or P2_out_bar <= 0.0:
        raise ValueError("Outlet pressures must stay positive after subtracting pressure drop.")

    # Stream 2 (CH4) duty from user-specified outlet temperature.
    h2_in = ch4_h_mass_from_yaml(yaml_path, T2_in_C, P2_in_bar)
    h2_out = ch4_h_mass_from_yaml(yaml_path, T2_out_C, P2_out_bar)
    Q_W = mdot_2 * (h2_out - h2_in)

    # Stream 1 (water) outlet from energy balance.
    h1_in = water_h_mass_real(T1_in_C, P1_in_bar)
    h1_out = h1_in - Q_W / mdot_1
    T1_out_C = water_T_from_hP_real(h1_out, P1_out_bar)

    # YAML H2O shown only to make the YAML usage visible in the app.
    h1_in_yaml_ref = h2o_h_mass_yaml_reference(yaml_path, T1_in_C, P1_in_bar)

    return {
        "mode": "Given Output 2 temperature",
        "method_note": (
            "Output 2 temperature is given by the user. The code uses it to compute "
            "CH4 enthalpy rise and heat duty, then solves the water outlet."
        ),
        "pressure_note": "Output pressure on each side is calculated from P_out = P_in - ΔP.",
        "mdot_1_kg_s": mdot_1,
        "mdot_2_kg_s": mdot_2,
        "T1_in_C": T1_in_C,
        "P1_in_bar": P1_in_bar,
        "T1_out_C": T1_out_C,
        "P1_out_bar": P1_out_bar,
        "T2_in_C": T2_in_C,
        "P2_in_bar": P2_in_bar,
        "T2_out_C": T2_out_C,
        "P2_out_bar": P2_out_bar,
        "Q_W": Q_W,
        "Q_kW": Q_W / 1e3,
        "Q_MW": Q_W / 1e6,
        "h1_in_real_kJ_kg": h1_in / 1e3,
        "h1_out_real_kJ_kg": h1_out / 1e3,
        "h1_in_yaml_ref_kJ_kg": h1_in_yaml_ref / 1e3,
        "h2_in_kJ_kg": h2_in / 1e3,
        "h2_out_kJ_kg": h2_out / 1e3,
        "dh2_kJ_kg": (h2_out - h2_in) / 1e3,
    }


def solve_mode_from_UA(
    yaml_path: Path,
    *,
    mdot_1: float,
    T1_in_C: float,
    P1_in_bar: float,
    mdot_2: float,
    T2_in_C: float,
    P2_in_bar: float,
    UA_W_per_K: float,
    dP1_bar: float,
    dP2_bar: float,
) -> Dict[str, float | str]:
    """
    Mode B: Output 2 temperature is unknown and is solved from UA.

    Equations:
      Q = mdot_1 * (h1_in - h1_out)
      Q = mdot_2 * (h2_out - h2_in)
      Q = UA * LMTD

    Unknown temperatures are recovered from Cantera property calls.
    """
    P1_out_bar = P1_in_bar - dP1_bar
    P2_out_bar = P2_in_bar - dP2_bar
    if P1_out_bar <= 0.0 or P2_out_bar <= 0.0:
        raise ValueError("Outlet pressures must stay positive after subtracting pressure drop.")
    if UA_W_per_K <= 0.0:
        raise ValueError("UA must be positive.")

    h1_in = water_h_mass_real(T1_in_C, P1_in_bar)
    h2_in = ch4_h_mass_from_yaml(yaml_path, T2_in_C, P2_in_bar)

    # Duty upper bound before temperature cross.
    h1_at_T2_in = water_h_mass_real(T2_in_C, P1_out_bar)
    h2_at_T1_in = ch4_h_mass_from_yaml(yaml_path, T1_in_C, P2_out_bar)
    Qmax_hot = mdot_1 * (h1_in - h1_at_T2_in)
    Qmax_cold = mdot_2 * (h2_at_T1_in - h2_in)
    Qmax = min(Qmax_hot, Qmax_cold)
    if Qmax <= 0.0:
        raise ValueError("No positive heat-transfer window exists for the given inlet states.")

    def residual(Q: float) -> float:
        h1_out = h1_in - Q / mdot_1
        h2_out = h2_in + Q / mdot_2
        T1_out_C = water_T_from_hP_real(h1_out, P1_out_bar)
        T2_out_C = ch4_T_from_hP_yaml(yaml_path, h2_out, P2_out_bar)
        dT1 = T1_in_C - T2_out_C
        dT2 = T1_out_C - T2_in_C
        lmtd = lmtd_countercurrent(dT1, dT2)
        return Q - UA_W_per_K * lmtd

    # Bisection for robust UA solve.
    a = 0.0
    b = 0.999999 * Qmax
    fa = residual(a)
    fb = residual(b)
    if fa * fb > 0.0:
        raise ValueError("Could not bracket the UA solution. Change UA or the flow rates.")

    for _ in range(200):
        m = 0.5 * (a + b)
        fm = residual(m)
        if abs(fm) < 1e-6 or abs(b - a) < 1e-6 * (1.0 + abs(m)):
            Q_W = m
            break
        if fa * fm < 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    else:
        Q_W = 0.5 * (a + b)

    h1_out = h1_in - Q_W / mdot_1
    h2_out = h2_in + Q_W / mdot_2
    T1_out_C = water_T_from_hP_real(h1_out, P1_out_bar)
    T2_out_C = ch4_T_from_hP_yaml(yaml_path, h2_out, P2_out_bar)

    dT1 = T1_in_C - T2_out_C
    dT2 = T1_out_C - T2_in_C
    lmtd = lmtd_countercurrent(dT1, dT2)
    h1_in_yaml_ref = h2o_h_mass_yaml_reference(yaml_path, T1_in_C, P1_in_bar)

    return {
        "mode": "Calculate Output 2 temperature from UA",
        "method_note": (
            "Output 2 temperature is solved from the coupled energy balance and the "
            "counter-current LMTD equation."
        ),
        "pressure_note": "Output pressure on each side is calculated from P_out = P_in - ΔP.",
        "mdot_1_kg_s": mdot_1,
        "mdot_2_kg_s": mdot_2,
        "T1_in_C": T1_in_C,
        "P1_in_bar": P1_in_bar,
        "T1_out_C": T1_out_C,
        "P1_out_bar": P1_out_bar,
        "T2_in_C": T2_in_C,
        "P2_in_bar": P2_in_bar,
        "T2_out_C": T2_out_C,
        "P2_out_bar": P2_out_bar,
        "Q_W": Q_W,
        "Q_kW": Q_W / 1e3,
        "Q_MW": Q_W / 1e6,
        "UA_W_per_K": UA_W_per_K,
        "LMTD_C": lmtd,
        "h1_in_real_kJ_kg": h1_in / 1e3,
        "h1_out_real_kJ_kg": h1_out / 1e3,
        "h1_in_yaml_ref_kJ_kg": h1_in_yaml_ref / 1e3,
        "h2_in_kJ_kg": h2_in / 1e3,
        "h2_out_kJ_kg": h2_out / 1e3,
        "dh2_kJ_kg": (h2_out - h2_in) / 1e3,
    }


# -----------------------------------------------------------------------------
# UI helpers that avoid pyarrow/pandas
# -----------------------------------------------------------------------------
def annotate_hex_image(image_path: Path) -> Image.Image:
    """Add port names and Heat_Exchanger title onto the image."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image.width * 3, image.height * 3))
    draw = ImageDraw.Draw(image)

    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:  # pragma: no cover
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    def boxed_text(xy: Tuple[int, int], text: str, font) -> None:
        x, y = xy
        bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=3)
        pad = 5
        draw.rounded_rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            radius=8,
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )
        draw.multiline_text((x, y), text, fill=(0, 0, 0), font=font, spacing=3)

    w, h = image.size
    boxed_text((10, h // 2 - 25), "Input 1\nWater in", font_big)
    boxed_text((w - 145, h // 2 - 25), "Output 1\nWater out", font_big)
    boxed_text((w // 2 - 55, h - 60), "Input 2\nCH4 in", font_big)
    boxed_text((w // 2 - 58, 5), "Output 2\nCH4 out", font_big)
    boxed_text((w // 2 - 90, h // 2 - 20), "Heat_Exchanger", font_small)
    return image


def fmt_num(value: float | str, digits: int = 4) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def build_port_rows(values: Dict[str, float | str]) -> List[List[str]]:
    return [
        ["Input 1", "Water", "Hot-side inlet", "Left", fmt_num(values["mdot_1_kg_s"]), fmt_num(values["T1_in_C"]), fmt_num(values["P1_in_bar"])],
        ["Output 1", "Water", "Hot-side outlet", "Right", fmt_num(values["mdot_1_kg_s"]), fmt_num(values["T1_out_C"]), fmt_num(values["P1_out_bar"])],
        ["Input 2", "CH4", "Cold-side inlet", "Bottom", fmt_num(values["mdot_2_kg_s"]), fmt_num(values["T2_in_C"]), fmt_num(values["P2_in_bar"])],
        ["Output 2", "CH4", "Cold-side outlet", "Top", fmt_num(values["mdot_2_kg_s"]), fmt_num(values["T2_out_C"]), fmt_num(values["P2_out_bar"])],
    ]


def html_table(headers: List[str], rows: List[List[str]]) -> str:
    head_html = "".join(
        f"<th style='border:1px solid #ddd;padding:8px;background:#f4f4f4;text-align:left'>{html.escape(h)}</th>"
        for h in headers
    )
    row_html = []
    for row in rows:
        cells = "".join(
            f"<td style='border:1px solid #ddd;padding:8px'>{html.escape(str(cell))}</td>"
            for cell in row
        )
        row_html.append(f"<tr>{cells}</tr>")
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:14px'>"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody></table>"
    )


def render_port_table(values: Dict[str, float | str]) -> None:
    headers = ["Port", "Fluid", "Role", "Image nozzle", "m_dot [kg/s]", "T [°C]", "P [bar]"]
    rows = build_port_rows(values)
    st.markdown(html_table(headers, rows), unsafe_allow_html=True)


def render_detail_table(values: Dict[str, float | str]) -> None:
    rows = [
        ["Water inlet enthalpy (real-fluid Cantera)", fmt_num(values["h1_in_real_kJ_kg"]), "kJ/kg"],
        ["Water outlet enthalpy (real-fluid Cantera)", fmt_num(values["h1_out_real_kJ_kg"]), "kJ/kg"],
        ["Water inlet enthalpy from YAML H2O (reference only)", fmt_num(values["h1_in_yaml_ref_kJ_kg"]), "kJ/kg"],
        ["CH4 inlet enthalpy from YAML", fmt_num(values["h2_in_kJ_kg"]), "kJ/kg"],
        ["CH4 outlet enthalpy from YAML", fmt_num(values["h2_out_kJ_kg"]), "kJ/kg"],
        ["CH4 enthalpy rise", fmt_num(values["dh2_kJ_kg"]), "kJ/kg"],
        ["Heat duty", fmt_num(values["Q_MW"]), "MW"],
    ]
    if "UA_W_per_K" in values:
        rows.append(["UA", fmt_num(values["UA_W_per_K"]), "W/K"])
        rows.append(["LMTD", fmt_num(values["LMTD_C"]), "°C"])
    st.markdown(html_table(["Quantity", "Value", "Unit"], rows), unsafe_allow_html=True)


def render_method_section() -> None:
    st.subheader("Method used")
    st.markdown(r"""
**Case A: given Output 2 temperature**

1. Set the CH4 inlet state with Cantera from \(T_{2,in}, P_{2,in}\).
2. Set the CH4 outlet state with Cantera from \(T_{2,out}, P_{2,out}\).
3. Compute CH4 heat gain.
4. Subtract the same duty from the water stream.
5. Recover water outlet temperature from water outlet enthalpy and pressure.

**Case B: calculate Output 2 temperature from UA**

1. Guess exchanger duty \(Q\).
2. Compute the corresponding outlet enthalpies of both streams.
3. Recover outlet temperatures using Cantera.
4. Compute LMTD.
5. Adjust \(Q\) until \(Q = UA\,\Delta T_{lm}\).
        """
    )

    st.subheader("Equations used")
    st.latex(r"P_{1,out} = P_{1,in} - \Delta P_1")
    st.latex(r"P_{2,out} = P_{2,in} - \Delta P_2")
    st.latex(r"Q = \dot m_2\,(h_{2,out} - h_{2,in})")
    st.latex(r"h_{1,out} = h_{1,in} - \frac{Q}{\dot m_1}")
    st.latex(r"Q = \dot m_1\,(h_{1,in} - h_{1,out})")
    st.markdown("For the UA mode, the code also solves:")
    st.latex(r"Q = UA\,\Delta T_{lm}")
    st.latex(r"\Delta T_{lm} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}")
    st.latex(r"\Delta T_1 = T_{1,in} - T_{2,out}")
    st.latex(r"\Delta T_2 = T_{1,out} - T_{2,in}")


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Heat_Exchanger - Cantera App ", layout="wide")
    st.title("Heat_Exchanger - Cantera Streamlit App")
    )

    if not HAS_CANTERA:
        st.error(
            "Cantera could not be imported in the current Python environment. "
            f"Import error: {CANTERA_IMPORT_ERROR}"
        )
        st.markdown(
            "**Windows/conda fix**  \\n"
            "`conda install -c conda-forge cantera`  \\n"
            "or  \\n"
            "`pip install cantera`"
        )
        st.stop()

    yaml_path = ensure_yaml_exists()

    with st.sidebar:
        st.header("Calculation mode")
        mode = st.radio(
            "How should Output 2 temperature be handled?",
            [
                "Given Output 2 temperature (your current case)",
                "Calculate Output 2 temperature from UA",
            ],
        )

        st.header("Stream 1 = Water")
        mdot_1 = st.number_input("Input 1 mass flow [kg/s]", min_value=0.0001, value=10.0, step=0.1)
        T1_in_C = st.number_input("Input 1 temperature [°C]", value=560.0, step=1.0)
        P1_in_bar = st.number_input("Input 1 pressure [bar]", min_value=0.001, value=140.0, step=1.0)
        dP1_bar = st.number_input("Water-side pressure drop ΔP1 [bar]", min_value=0.0, value=0.0, step=0.1)

        st.header("Stream 2 = Methane")
        mdot_2 = st.number_input("Input 2 mass flow [kg/s]", min_value=0.0001, value=1.0, step=0.1)
        T2_in_C = st.number_input("Input 2 temperature [°C]", value=20.0, step=1.0)
        P2_in_bar = st.number_input("Input 2 pressure [bar]", min_value=0.001, value=5.0, step=0.1)
        dP2_bar = st.number_input("Methane-side pressure drop ΔP2 [bar]", min_value=0.0, value=0.0, step=0.1)

        if mode == "Given Output 2 temperature (your current case)":
            T2_out_C = st.number_input("Output 2 temperature [°C]", value=200.0, step=1.0)
            UA_W_per_K = None
        else:
            T2_out_C = None
            UA_W_per_K = st.number_input("UA [W/K]", min_value=0.001, value=20000.0, step=100.0)

    col_img, col_info = st.columns([1.15, 1.3])

    with col_img:
        if IMAGE_PATH.exists():
            st.image(annotate_hex_image(IMAGE_PATH), caption="Heat_Exchanger")
        else:
            st.warning(f"Image not found: {IMAGE_PATH}")

    with col_info:
        st.subheader("Port naming used in this app")
        st.markdown(
            "- **Input 1** = water inlet = left nozzle  \n"
            "- **Output 1** = water outlet = right nozzle  \n"
            "- **Input 2** = CH4 inlet = bottom nozzle  \n"
            "- **Output 2** = CH4 outlet = top nozzle"
        )
        st.info(
            "The nozzle names are only the UI naming convention. The thermodynamic calculation depends on the stream data."
        )
        st.subheader("How Output 2 is handled")
        if mode == "Given Output 2 temperature (your current case)":
            st.write(
                "In this mode, Output 2 temperature is a user input. The app uses it to calculate CH4 enthalpy rise and exchanger duty."
            )
        else:
            st.write(
                "In this mode, Output 2 temperature is calculated from the energy balance and the counter-current LMTD equation."
            )
        st.write("On both streams, outlet pressure is calculated from P_out = P_in - ΔP.")

    try:
        if mode == "Given Output 2 temperature (your current case)":
            result = solve_mode_given_output2(
                yaml_path,
                mdot_1=mdot_1,
                T1_in_C=T1_in_C,
                P1_in_bar=P1_in_bar,
                mdot_2=mdot_2,
                T2_in_C=T2_in_C,
                P2_in_bar=P2_in_bar,
                T2_out_C=float(T2_out_C),
                dP1_bar=dP1_bar,
                dP2_bar=dP2_bar,
            )
        else:
            result = solve_mode_from_UA(
                yaml_path,
                mdot_1=mdot_1,
                T1_in_C=T1_in_C,
                P1_in_bar=P1_in_bar,
                mdot_2=mdot_2,
                T2_in_C=T2_in_C,
                P2_in_bar=P2_in_bar,
                UA_W_per_K=float(UA_W_per_K),
                dP1_bar=dP1_bar,
                dP2_bar=dP2_bar,
            )
    except Exception as exc:
        st.error(f"Calculation failed: {exc}")
        st.stop()

    st.subheader("Inputs and outputs")
    render_port_table(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Heat duty Q [kW]", fmt_num(result["Q_kW"], 3))
    c2.metric("Output 1 temperature [°C]", fmt_num(result["T1_out_C"], 3))
    c3.metric("Output 2 pressure [bar]", fmt_num(result["P2_out_bar"], 3))

    st.subheader("Explanation of the calculated result")
    st.write(result["method_note"])
    st.write(result["pressure_note"])

    st.subheader("Thermodynamic details from Cantera")
    render_detail_table(result)

    with st.expander("Method and equations"):
        render_method_section()

    with st.expander("Where Cantera is used in the code"):
        st.code(
            """
# CH4 from YAML
h2_in = ch4_h_mass_from_yaml(yaml_path, T2_in_C, P2_in_bar)
h2_out = ch4_h_mass_from_yaml(yaml_path, T2_out_C, P2_out_bar)

# Real-fluid water
h1_in = water_h_mass_real(T1_in_C, P1_in_bar)
T1_out_C = water_T_from_hP_real(h1_out, P1_out_bar)

# In UA mode, CH4 outlet temperature comes from an enthalpy inversion
T2_out_C = ch4_T_from_hP_yaml(yaml_path, h2_out, P2_out_bar)
            """,
            language="python",
        )

    with st.expander("Embedded YAML written by this app"):
        st.code(EMBEDDED_YAML, language="yaml")
        st.caption(f"Saved to: {yaml_path}")


if __name__ == "__main__":
    main()

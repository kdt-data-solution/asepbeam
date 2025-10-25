# app.py — Beam Analysis & RC Design with Table, Validation, Deflection, and Row Add/Remove
from __future__ import annotations

import copy
import math
import numpy as np
import sympy as sp
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as shp
from matplotlib.lines import Line2D

import streamlit as st

# -------------------------
# Units
# -------------------------
m, to_m = [1, 1]
mm, to_mm = [1e-3, 1e3]
N, to_N = [1, 1]
kN, to_kN = [1e3, 1e-3]
kPa, to_kPa = [1e3, 1e-3]
MPa, to_MPa = [1e6, 1e-6]
GPa, to_GPa = [1e9, 1e-9]

# -------------------------
# Helpers / formatting
# -------------------------
sgn = lambda x, pow: 0 if x < 0 or pow < 0 else x**pow

def _fmt(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    elif abs(v) >= 100:
        return f"{v:,.1f}"
    else:
        return f"{v:,.2f}"

# -------------------------
# Beam plot with annotations + legend
# -------------------------
def plot_beam(L, x_sup1, x_sup2, loads, scale=1):
    fig, ax = plt.subplots()

    # Beam and supports
    ax.plot([0, L], [0, 0], color='black', linewidth=5)
    ax.add_artist(shp.Polygon([[x_sup1, 0], [x_sup1+0.25*m, -0.5*m], [x_sup1-0.25*m, -0.5*m]], color='black'))
    ax.add_artist(shp.Circle([x_sup2, -0.25*m], radius=0.25*m, color='black'))

    for load in loads:
        case = load[1] if len(load) > 1 else "dead"
        color = 'red' if case == "dead" else 'blue'

        if load[0] == "point":
            # point load (N) at x
            P = float(load[-2])
            x0 = float(load[-1])
            y0 = -P*scale if P < 0 else 0.0
            dy = P*scale
            ax.add_artist(shp.Arrow(x0, y0, 0, dy, color=color, width=0.3))
            # label to the RIGHT of the arrow
            txt = f"{_fmt(P/kN)} kN"
            ax.text(x0 + 0.15*m, y0 + (0.05 if dy >= 0 else -0.05), txt,
                    color=color, ha='left', va='bottom' if dy>=0 else 'top', fontsize=9)

        elif load[0] == "uniform":
            # uniform load (N/m) from a to b
            w = float(load[-2])
            a, b = map(float, load[-1])
            if w < 0:
                ax.add_artist(shp.Arrow(a, -w*scale, 0, w*scale, color=color, width=0.3))
                ax.add_artist(shp.Arrow(b, -w*scale, 0, w*scale, color=color, width=0.3))
                ax.plot([a, b], [-w*scale, -w*scale], color=color, linewidth=3)
            elif w > 0:
                ax.add_artist(shp.Arrow(a, 0, 0, w*scale, color=color, width=0.3))
                ax.add_artist(shp.Arrow(b, 0, 0, w*scale, color=color, width=0.3))
                ax.plot([a, b], [w*scale, w*scale], color=color, linewidth=3)

            # annotation: always above x-axis; live slightly higher than dead
            xm = 0.5*(a+b)
            txt = f"{_fmt(w/(kN/m))} kN/m"
            y_base = 1.0
            y_txt = y_base - 0.2 if case == 'dead' else y_base
            ax.text(xm, y_txt, txt, color=color, ha='center', va='bottom', fontsize=9)

        elif load[0] == "moment":
            M = float(load[-2]); x0 = float(load[-1])
            if M > 0:
                ax.add_artist(shp.FancyArrowPatch((x0, -0.5*m), (x0, 0.5*m),
                                                  connectionstyle="arc3, rad=1",
                                                  arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                  color=color))
            elif M < 0:
                ax.add_artist(shp.FancyArrowPatch((x0, 0.5*m), (x0, -0.5*m),
                                                  connectionstyle="arc3, rad=1",
                                                  arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                  color=color))
            txt = f"{_fmt(M/(kN*m))} kN·m"
            ax.text(x0, 0.6*m, txt, color=color, ha='center', va='bottom', fontsize=9)

    ax.axis("equal")
    ax.set_xlabel(r"$x$ (m)")
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Dead load'),
        Line2D([0], [0], color='blue', lw=3, label='Live load'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    return fig

# -------------------------
# Load combinations
# -------------------------
def factored_loads(loads):
    d1 = copy.deepcopy(loads); d2 = copy.deepcopy(loads)
    l1 = copy.deepcopy(loads); l2 = copy.deepcopy(loads)

    asd1, asd2, lrfd1, lrfd2 = [], [], [], []

    for load in d1:
        if load[1] == "dead":
            load.pop(1); asd1.append(load)

    for load in l1:
        if load[1] == "dead":
            load[-2] *= 1.4
            load.pop(1); lrfd1.append(load)

    for load in d2:
        load.pop(1); asd2.append(load)

    for load in l2:
        if load[1] == "dead":
            load[-2] *= 1.2
        elif load[1] == "live":
            load[-2] *= 1.6
        load.pop(1); lrfd2.append(load)

    return (asd1, asd2, lrfd1, lrfd2)

# -------------------------
# Reactions
# -------------------------
def support_reactions(factored_loads_list, x_sup1, x_sup2):
    reactions_sup1, reactions_sup2 = [], []
    for factored_load in factored_loads_list:
        moment_arm = [
            (load[-1][0] + load[-1][1])/2 - x_sup1 if load[0] == "uniform" else load[-1] - x_sup1
            for load in factored_load
        ]
        sum_moment = 0.0; sum_force = 0.0
        for i, load in enumerate(factored_load):
            if load[0] == "moment":
                sum_moment += load[-2]
            elif load[0] == "point":
                sum_moment += load[-2] * moment_arm[i]
                sum_force += load[-2]
            elif load[0] == "uniform":
                span = load[-1][1] - load[-1][0]
                sum_moment += load[-2] * span * moment_arm[i]
                sum_force += load[-2] * span

        r2 = -sum_moment / (x_sup2 - x_sup1)
        r1 = -sum_force - r2
        reactions_sup1.append(r1); reactions_sup2.append(r2)
    return (reactions_sup1, reactions_sup2)

def updated_factored_loads(factored_loads_list, reactions_sup1, x_sup1, reactions_sup2, x_sup2):
    flc = copy.deepcopy(factored_loads_list)
    for factored_load, r1, r2 in zip(flc, reactions_sup1, reactions_sup2):
        factored_load.append(['point', r1, x_sup1])
        factored_load.append(['point', r2, x_sup2])
    return flc

# -------------------------
# Shear & Moment
# -------------------------
def shear(load_, x):
    V = 0.0
    for load in load_:
        if load[0] == "point":     V += load[-2] * sgn(x - load[-1], 0)
        elif load[0] == "moment":  V += -load[-2] * sgn(x - load[-1], -1)
        elif load[0] == "uniform": V += load[-2]*sgn(x - load[-1][0], 1) - load[-2]*sgn(x - load[-1][1], 1)
    return V

def moment(load_, x):
    M = 0.0
    for load in load_:
        if load[0] == "point":     M += load[-2] * sgn(x - load[-1], 1)
        elif load[0] == "moment":  M += -load[-2] * sgn(x - load[-1], 0)
        elif load[0] == "uniform": M += load[-2]*sgn(x - load[-1][0], 2)/2 - load[-2]*sgn(x - load[-1][1], 2)/2
    return M

def factored_shear(factored_loads_list, L, n_points=100):
    x = np.linspace(0, L, n_points)
    return [[shear(fl, xi) for xi in x] for fl in factored_loads_list]

def factored_moment(factored_loads_list, L, n_points=100):
    x = np.linspace(0, L, n_points)
    return [[moment(fl, xi) for xi in x] for fl in factored_loads_list]

def plot_combo_diagrams(updated_factored_loads_list, L, n_points=100, combo_idx=0):
    x = np.linspace(0, L, n_points)
    shear_all = factored_shear(updated_factored_loads_list, L, n_points=n_points)
    moment_all = factored_moment(updated_factored_loads_list, L, n_points=n_points)
    combo_idx = int(max(0, min(3, combo_idx)))
    V = shear_all[combo_idx] if len(shear_all) > combo_idx else [0]*len(x)
    M = moment_all[combo_idx] if len(moment_all) > combo_idx else [0]*len(x)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(x, V, 'r-', linewidth=2); ax[0].axhline(0, color='black', linewidth=2)
    ax[0].set_xlabel(r"$x$ (m)"); ax[0].set_ylabel("Shear (N)"); ax[0].set_title("Shear")

    ax[1].plot(x, M, 'r-', linewidth=2); ax[1].axhline(0, color='black', linewidth=2)
    ax[1].set_xlabel(r"$x$ (m)"); ax[1].set_ylabel("Bending moment (N·m)"); ax[1].set_title("Moment")
    fig.tight_layout()
    return fig

# -------------------------
# RC design
# -------------------------
def rc_design_beam_size(Mu_max, fcp, fy, cc, db, dst, aspectratio=2):
    b_sym = sp.Symbol("b")
    phi = 0.90
    beta1 = max(min(0.85 - 0.05/7*(fcp - 28), 0.85), 0.65)
    rho5 = 3/8 * 0.85*fcp*beta1/fy
    omega5 = rho5*fy/fcp
    eqn = sp.Eq(Mu_max, phi*b_sym*(aspectratio*b_sym - cc - dst - db/2)**2*fcp*omega5*(1 - 10*omega5/17))
    b_sol = sp.solve(eqn, b_sym)[0]
    d = aspectratio*b_sol
    h_sol = d + cc + dst + db/2
    b = max(math.ceil(float(b_sol)/50)*50, 200)
    h = max(math.ceil(float(h_sol)/50)*50, 200)
    return int(b), int(h)

def rc_design_beam_main(bending_moments, b, h, fcp, fy, cc, db, dst):
    As_req, Asp_req = [], []
    phi = 0.90
    d = h - cc - dst - db/2
    m = fy/(0.85*fcp)
    rho_min = max(1.4, 0.25*math.sqrt(fcp))/fy
    for Mu in bending_moments:
        Mu_ = Mu if Mu >= 0 else abs(Mu)
        Rn = Mu_*to_N*to_mm/(phi*b*d**2)
        rho = 1/m*(1 - math.sqrt(1 - 2*m*Rn/fy))
        rho = max(rho, rho_min)
        As = rho*b*d
        As_req.append(As if Mu >= 0 else 0)
        Asp_req.append(0 if Mu >= 0 else As)
    return (As_req, Asp_req)

def plot_main_rebar(L, As_req, Asp_req):
    x = np.linspace(0, L, len(As_req))
    fig, ax = plt.subplots(figsize=(6, 3.2))  # smaller
    ax.plot(x, -1*np.array(As_req, dtype='float'), 'r-', linewidth=2)
    ax.plot(x, Asp_req, color='blue', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel('As (mm^2)')
    ax.legend(["Bottom bars", "Top bars"])
    return fig

# -------------------------
# Deflection
# -------------------------
c1, c2 = sp.symbols('C1 C2')

def deflection_func(load_, x, E, b_m, h_m, k_mod=0.35):
    I = (1/12) * b_m * (h_m**3)
    deflection_ = 0
    for load in load_:
        if load[0] == "point":
            deflection_ += (load[-2]*sgn(x - load[-1], 3)/6 + c1*x + c2)/(k_mod*E*I)
        elif load[0] == "moment":
            deflection_ += (-load[-2]*sgn(x - load[-1], 2)/2 + c1*x + c2)/(k_mod*E*I)
        elif load[0] == "uniform":
            deflection_ += (load[-2]*sgn(x - load[-1][0], 4)/24 - load[-2]*sgn(x - load[-1][1], 4)/24 + c1*x + c2)/(k_mod*E*I)
    return deflection_

def plot_deflection(load_, L, x_sup1, x_sup2, fcp_MPa, b_mm, h_mm, n_points=200, k_mod=0.35):
    Ec = 4700*math.sqrt(fcp_MPa) * MPa
    b_m = b_mm * mm; h_m = h_mm * mm
    eqn1 = deflection_func(load_, float(x_sup1), Ec, b_m, h_m, k_mod=k_mod)
    eqn2 = deflection_func(load_, float(x_sup2), Ec, b_m, h_m, k_mod=k_mod)
    res = sp.solve([eqn1, eqn2], [c1, c2])
    def_eqn = lambda x: deflection_func(load_, x, Ec, b_m, h_m, k_mod=k_mod).subs(res)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.linspace(0, L, n_points)
    y = [float(def_eqn(xi)) for xi in x]
    ax.plot(x, y, 'r-', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel('Deflection (m)')
    fig.tight_layout()
    return fig

# -------------------------
# Default test beam #2
# -------------------------
L_2 = 10*m
x_sup1_2 = 0*m
x_sup2_2 = L_2 - 3*m

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Beam Analysis & RC Design", layout="wide")
    st.title("Beam Analysis & RC Design (ASD/LRFD)")

    st.sidebar.header("Inputs")
    preset = st.sidebar.selectbox("Preset", ["Custom via table (recommended)", "Test beam #2"], index=0)

    # Geometry
    L = st.sidebar.number_input("Beam length L (m)", value=float(L_2), min_value=0.1, step=0.1)
    x_sup1 = st.sidebar.number_input("x support 1 (m)", value=float(x_sup1_2), min_value=0.0, max_value=L, step=0.1)
    x_sup2 = st.sidebar.number_input("x support 2 (m)", value=float(x_sup2_2), min_value=0.0, max_value=L, step=0.1)

    st.sidebar.divider()
    scale = st.sidebar.slider("Load arrow scale (viz only)", min_value=1e-6, max_value=5e-5,
                              value=1.5e-5, step=1e-6, format="%.1e")
    n_points = st.sidebar.slider("Discretization (n points)", 10, 2000, 500, 10)

    st.sidebar.divider()
    st.sidebar.subheader("Concrete & Steel")
    fcp = st.sidebar.number_input("f'c (MPa)", value=34.5, min_value=10.0, step=0.5)
    fy = st.sidebar.number_input("fy (MPa)", value=415.0, min_value=200.0, step=5.0)
    cc = st.sidebar.number_input("clear cover cc (mm)", value=40.0, min_value=20.0, step=5.0)
    db = st.sidebar.number_input("bar diameter db (mm)", value=20.0, min_value=8.0, step=1.0)
    dst = st.sidebar.number_input("stirrup dia dst (mm)", value=12.0, min_value=6.0, step=1.0)
    aspect = st.sidebar.slider("d:b aspect ratio", min_value=1.2, max_value=2.5, value=1.5, step=0.1)

    # -------------------------
    # Load table with Add/Remove
    # -------------------------
    st.subheader("Load input table (with magnitudes)")
    st.caption("Magnitudes: use kN for point, kN/m for uniform. Downward is negative.")

    default_rows = [
        {"case": "dead", "type": "uniform", "mag": -20.0, "x_point": None, "x_start": (2*L/3), "x_end": L},
        {"case": "live", "type": "uniform", "mag": -30.0, "x_point": None, "x_start": (2*L/3), "x_end": L},
        {"case": "dead", "type": "point",   "mag": -120.0, "x_point": (L/3),  "x_start": None, "x_end": None},
        {"case": "live", "type": "point",   "mag": -250.0, "x_point": (L/3),  "x_start": None, "x_end": None},
    ]
    if "loads_df" not in st.session_state or preset == "Test beam #2":
        st.session_state.loads_df = pd.DataFrame(default_rows)

    # Ensure delete checkbox column exists
    if "del" not in st.session_state.loads_df.columns:
        st.session_state.loads_df["del"] = False

    # Row control buttons
    c_add, c_rem = st.columns(2)
    add_clicked = c_add.button("➕ Add row", use_container_width=True)
    rem_clicked = c_rem.button("🗑 Remove selected", use_container_width=True)

    if add_clicked:
        new_row = {"case": "dead", "type": "point", "mag": -10.0,
                   "x_point": 0.0, "x_start": None, "x_end": None, "del": False}
        st.session_state.loads_df = pd.concat(
            [st.session_state.loads_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    edited_df = st.data_editor(
        st.session_state.loads_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "case": st.column_config.SelectboxColumn("Load case", options=["dead", "live"], width=100),
            "type": st.column_config.SelectboxColumn("Load type", options=["uniform", "point"], width=120),
            "mag":  st.column_config.NumberColumn("Magnitude (kN or kN/m)",
                                                  help="kN for point, kN/m for uniform; sign matters (downward negative)",
                                                  step=0.5, format="%.3f"),
            "x_point": st.column_config.NumberColumn("Point load location (m)",
                                                     min_value=0.0, max_value=L, step=0.01, format="%.3f"),
            "x_start": st.column_config.NumberColumn("Start load location (m)",
                                                     min_value=0.0, max_value=L, step=0.01, format="%.3f"),
            "x_end":   st.column_config.NumberColumn("End load location (m)",
                                                     min_value=0.0, max_value=L, step=0.01, format="%.3f"),
            "del": st.column_config.CheckboxColumn("Del",
                                                   help="Check rows to remove, then press the trash button.",
                                                   default=False, width=60),
        },
        hide_index=True,
        key="load_table_editor",
    )

    # Handle removal & persist edits
    if rem_clicked and "del" in edited_df.columns:
        keep_mask = ~edited_df["del"].fillna(False)
        st.session_state.loads_df = edited_df[keep_mask].drop(columns=["del"]).reset_index(drop=True)
        st.session_state.loads_df["del"] = False
        st.rerun()
    else:
        st.session_state.loads_df = edited_df

    # -------------------------
    # Validations (supports, loads)
    # -------------------------
    def build_loads_from_table_validated(df: pd.DataFrame, L: float):
        loads = []; errors = []
        for idx, row in df.iterrows():
            row_no = idx + 1
            case = str(row.get("case", "")).strip().lower()
            ltype = str(row.get("type", "")).strip().lower()
            mag = row.get("mag")

            # magnitude
            if mag is None or (isinstance(mag, float) and math.isnan(mag)):
                errors.append(f"Row {row_no}: magnitude is required.")
                continue
            try:
                mag = float(mag)
            except Exception:
                errors.append(f"Row {row_no}: magnitude must be numeric.")
                continue
            if mag >= 0:
                errors.append(f"Row {row_no}: magnitude must be negative (downward).")
                continue

            if ltype == "point":
                x = row.get("x_point")
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    errors.append(f"Row {row_no}: point load location is required.")
                    continue
                try:
                    x = float(x)
                except Exception:
                    errors.append(f"Row {row_no}: point location must be numeric.")
                    continue
                if not (0.0 <= x <= L):
                    errors.append(f"Row {row_no}: point location {x} m is outside [0, {L}] m.")
                    continue
                loads.append(["point", case, mag * kN, x])

            elif ltype == "uniform":
                xs = row.get("x_start"); xe = row.get("x_end")
                if xs is None or (isinstance(xs, float) and math.isnan(xs)) or \
                   xe is None or (isinstance(xe, float) and math.isnan(xe)):
                    errors.append(f"Row {row_no}: start and end locations required for uniform loads.")
                    continue
                try:
                    a = float(xs); b = float(xe)
                except Exception:
                    errors.append(f"Row {row_no}: start/end must be numeric.")
                    continue
                if a > b: a, b = b, a
                if a < 0.0 or b > L:
                    errors.append(f"Row {row_no}: uniform range [{a}, {b}] m must be within [0, {L}] m.")
                    continue
                if abs(b - a) <= 0:
                    errors.append(f"Row {row_no}: uniform range start and end cannot be equal.")
                    continue
                loads.append(["uniform", case, mag * kN / m, (a, b)])

            else:
                errors.append(f"Row {row_no}: unknown load type '{ltype}'.")
        return loads, errors

    if preset == "Test beam #2":
        loads, load_errors = build_loads_from_table_validated(pd.DataFrame(default_rows), L)
    else:
        loads, load_errors = build_loads_from_table_validated(st.session_state.loads_df, L)

    # Support validation
    errors = []
    try:
        xs1 = float(x_sup1); xs2 = float(x_sup2)
    except Exception:
        errors.append("Support locations must be numeric.")
        xs1, xs2 = 0.0, L
    if not (0.0 <= xs1 <= L) or not (0.0 <= xs2 <= L):
        errors.append("Support locations must be within the beam length [0, L].")
    if xs1 == xs2:
        errors.append("Support locations must not coincide.")
    if abs(xs2 - xs1) > L:
        errors.append("Support distance must not exceed beam length.")

    errors.extend(load_errors)

    if errors:
        st.error("\n".join(f"• {e}" for e in errors))
        return

    # -------------------------
    # Analysis & plotting
    # -------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Beam & Loads")
        fig_beam = plot_beam(L, xs1, xs2, loads, scale=scale)
        st.pyplot(fig_beam, use_container_width=True)

    # Factoring, reactions
    asd1, asd2, lrfd1, lrfd2 = factored_loads(loads)
    r1_list, r2_list = support_reactions((asd1, asd2, lrfd1, lrfd2), xs1, xs2)
    upd = updated_factored_loads((asd1, asd2, lrfd1, lrfd2), r1_list, xs1, r2_list, xs2)

    with col2:
        st.subheader("Support Reactions (by combo)")
        df_rxn = pd.DataFrame({
            'Combo': ["ASD1", "ASD2", "LRFD1", "LRFD2"],
            'R1 (N)': r1_list,
            'R2 (N)': r2_list,
        })
        st.dataframe(df_rxn, use_container_width=True)

    st.divider()
    st.subheader("Shear & Moment (select combination)")
    combo_names = ["ASD1", "ASD2", "LRFD1", "LRFD2"]
    selected = st.selectbox("Load combination", combo_names, index=2)
    combo_idx = combo_names.index(selected)
    fig_SM = plot_combo_diagrams(upd, L, n_points=n_points, combo_idx=combo_idx)
    st.pyplot(fig_SM, use_container_width=True)

    # Deflection below shear & moment
    st.subheader("Deflection (selected combination)")
    # Use designed section, so compute LRFD envelope etc. first (but we can plot deflection with current combo)
    lrfd_moms = factored_moment(upd, L, n_points=max(n_points, 200))
    lrfd_only = lrfd_moms[2:] if len(lrfd_moms) >= 4 else []
    moment_envelope = [max(lrfd_only[0][i], lrfd_only[1][i]) for i in range(len(lrfd_only[0]))] if lrfd_only else []
    Mu_max = float(np.max(np.abs(np.array(moment_envelope, dtype='float')))) if moment_envelope else 0.0

    # If no loads, stop
    if Mu_max <= 0:
        st.info("Define at least one nonzero load to perform RC design and deflection.")
        return

    # RC sizing
    b, h = rc_design_beam_size(Mu_max*to_N*to_mm, fcp, fy, cc, db, dst, aspectratio=aspect)
    fig_defl = plot_deflection(upd[combo_idx], L, xs1, xs2, fcp_MPa=float(fcp), b_mm=int(b), h_mm=int(h),
                               n_points=max(200, n_points))
    st.pyplot(fig_defl, use_container_width=True)

    # RC main rebar
    As_req, Asp_req = rc_design_beam_main(moment_envelope, b, h, fcp, fy, cc, db, dst)
    st.subheader("RC Beam Design (Main Reinforcement)")
    colA, colB, colC = st.columns([1, 1, 1])
    colA.metric("Mu,max (N·m)", f"{float(Mu_max):,.0f}")
    colB.metric("b (mm)", f"{int(b):,}")
    colC.metric("h (mm)", f"{int(h):,}")

    fig_As = plot_main_rebar(L, As_req, Asp_req)
    st.pyplot(fig_As, use_container_width=True)
    df_as = pd.DataFrame({
        'x (m)': np.linspace(0, L, len(As_req)),
        'As_bottom (mm^2)': np.array(As_req, dtype='float'),
        'As_top (mm^2)': np.array(Asp_req, dtype='float'),
    })
    st.dataframe(df_as, use_container_width=True)

if __name__ == "__main__":
    main()

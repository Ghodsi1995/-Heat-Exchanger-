# -Heat-Exchanger-
A sample heat exchanger with water and methane as inputs
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

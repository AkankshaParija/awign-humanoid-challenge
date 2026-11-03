"""Single source of truth for measured object dimensions (metres).

Everything in sim/ and scripts/ that needs cup or sticky-note geometry
reads these constants instead of hardcoding numbers. Measured by hand
with a ruler against the real cup and sticky note used for filming.
"""

# --- cup: tapered, wider at the top than the bottom ---
CUP_TOP_RADIUS = 0.0375      # 7.5 cm diameter
CUP_BOTTOM_RADIUS = 0.0225   # 4.5 cm diameter
CUP_HEIGHT = 0.08             # 8 cm
CUP_MASS = 0.12               # kg, empty paper/plastic cup estimate
CUP_TAPER_SEGMENTS = 6        # stacked cylinders approximating the taper

# --- sticky note: thin flat square, the pick-and-place target ---
STICKY_NOTE_SIDE = 0.075     # 7.5 cm x 7.5 cm
STICKY_NOTE_THICKNESS = 0.001  # 0.1 cm

# --- table ---
TABLE_HEIGHT = 0.30           # tabletop surface height off the floor
TABLE_TOP_HALF_THICKNESS = 0.015

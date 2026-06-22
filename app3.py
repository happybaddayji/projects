import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Airfoil Lift & Drag Visualizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .stall-warning {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ef4444;
        color: #fecaca;
    }
    .stall-warning b, .stall-warning strong {
        color: #ffffff;
    }
    .info-box {
        background: #1e3a5f;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        margin: 1rem 0;
        color: #e0f2fe;
    }
    .info-box b, .info-box strong {
        color: #60a5fa;
    }
    .theory-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #e2e8f0;
    }
    .quiz-option {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    .disclaimer {
        background: #78350f;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f59e0b;
        color: #fef3c7;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    .disclaimer b, .disclaimer strong {
        color: #fbbf24;
    }
    /* Fix for dark theme text visibility */
    .stMarkdown, .stText {
        color: #e2e8f0;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #f1f5f9;
    }
    div[data-testid="stExpander"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary {
        color: #e2e8f0;
    }
    .element-container iframe {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)
# ============================================
# AIRFOIL DATA
# ============================================
PREDEFINED_AIRFOILS = {
    "NACA 0012": {
        "name": "NACA 0012",
        "type": "Symmetric",
        "camber_description": "No camber (symmetric)",
        "max_thickness": 0.12,
        "max_camber": 0.0,
        "zero_lift_angle": 0,
        "stall_angle": 15,
        "base_drag": 0.006,
        "description": "A symmetric airfoil with 12% thickness. Commonly used for tail sections and aerobatic aircraft. Generates zero lift at zero angle of attack since it has no camber.",
        "coordinates": [
            (1.0, 0.0013), (0.95, 0.0095), (0.9, 0.0171), (0.8, 0.0303),
            (0.7, 0.0411), (0.6, 0.0492), (0.5, 0.0540), (0.4, 0.0545),
            (0.3, 0.0494), (0.2, 0.0388), (0.1, 0.0263), (0.05, 0.0194),
            (0.025, 0.0141), (0.0, 0.0), (0.025, -0.0141), (0.05, -0.0194),
            (0.1, -0.0263), (0.2, -0.0388), (0.3, -0.0494), (0.4, -0.0545),
            (0.5, -0.0540), (0.6, -0.0492), (0.7, -0.0411), (0.8, -0.0303),
            (0.9, -0.0171), (0.95, -0.0095), (1.0, -0.0013)
        ]
    },
    "NACA 2412": {
        "name": "NACA 2412",
        "type": "Cambered",
        "camber_description": "2% camber at 40% chord",
        "max_thickness": 0.12,
        "max_camber": 0.02,
        "zero_lift_angle": -2,
        "stall_angle": 16,
        "base_drag": 0.007,
        "description": "A mildly cambered airfoil with 2% maximum camber at 40% chord and 12% thickness. Used extensively in general aviation. Provides good lift with moderate drag.",
        "coordinates": [
            (1.0, 0.0013), (0.95, 0.0114), (0.9, 0.0208), (0.8, 0.0375),
            (0.7, 0.0510), (0.6, 0.0612), (0.5, 0.0672), (0.4, 0.0679),
            (0.3, 0.0614), (0.2, 0.0483), (0.1, 0.0327), (0.05, 0.0253),
            (0.025, 0.0187), (0.0, 0.0), (0.025, -0.0119), (0.05, -0.0160),
            (0.1, -0.0207), (0.2, -0.0260), (0.3, -0.0280), (0.4, -0.0267),
            (0.5, -0.0230), (0.6, -0.0178), (0.7, -0.0125), (0.8, -0.0079),
            (0.9, -0.0040), (0.95, -0.0022), (1.0, -0.0013)
        ]
    },
    "NACA 4412": {
        "name": "NACA 4412",
        "type": "Cambered",
        "camber_description": "4% camber at 40% chord",
        "max_thickness": 0.12,
        "max_camber": 0.04,
        "zero_lift_angle": -4,
        "stall_angle": 14,
        "base_drag": 0.008,
        "description": "A moderately cambered airfoil with 4% maximum camber at 40% chord. Provides higher lift coefficient than NACA 2412 at the cost of slightly more drag. Used in utility aircraft.",
        "coordinates": [
            (1.0, 0.0013), (0.95, 0.0136), (0.9, 0.0252), (0.8, 0.0453),
            (0.7, 0.0616), (0.6, 0.0738), (0.5, 0.0809), (0.4, 0.0816),
            (0.3, 0.0737), (0.2, 0.0579), (0.1, 0.0397), (0.05, 0.0313),
            (0.025, 0.0233), (0.0, 0.0), (0.025, -0.0098), (0.05, -0.0126),
            (0.1, -0.0155), (0.2, -0.0173), (0.3, -0.0168), (0.4, -0.0146),
            (0.5, -0.0114), (0.6, -0.0082), (0.7, -0.0052), (0.8, -0.0029),
            (0.9, -0.0013), (0.95, -0.0008), (1.0, -0.0013)
        ]
    },
    "Clark Y": {
        "name": "Clark Y",
        "type": "Cambered",
        "camber_description": "3.5% camber, flat bottom",
        "max_thickness": 0.117,
        "max_camber": 0.035,
        "zero_lift_angle": -3.5,
        "stall_angle": 15,
        "base_drag": 0.0075,
        "description": "A classic flat-bottom airfoil designed in the 1920s. Known for its simplicity and good performance. The flat lower surface makes it easy to build in model aircraft.",
        "coordinates": [
            (1.0, 0.0010), (0.95, 0.0120), (0.9, 0.0222), (0.8, 0.0408),
            (0.7, 0.0562), (0.6, 0.0682), (0.5, 0.0756), (0.4, 0.0770),
            (0.3, 0.0710), (0.2, 0.0568), (0.1, 0.0380), (0.05, 0.0280),
            (0.025, 0.0205), (0.0, 0.0), (0.025, -0.0050), (0.05, -0.0060),
            (0.1, -0.0060), (0.2, -0.0040), (0.3, -0.0020), (0.4, -0.0005),
            (0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.8, 0.0),
            (0.9, 0.0), (0.95, 0.0), (1.0, 0.0)
        ]
    },
    "Selig S1223": {
        "name": "Selig S1223",
        "type": "High-Lift",
        "camber_description": "8.6% camber, high-lift design",
        "max_thickness": 0.121,
        "max_camber": 0.086,
        "zero_lift_angle": -7,
        "stall_angle": 13,
        "base_drag": 0.012,
        "description": "A high-lift airfoil designed by Michael Selig. Produces very high lift coefficients at low Reynolds numbers. Used in low-speed UAVs and model aircraft. Has higher drag due to aggressive camber.",
        "coordinates": [
            (1.0, 0.0005), (0.95, 0.0170), (0.9, 0.0340), (0.8, 0.0640),
            (0.7, 0.0890), (0.6, 0.1080), (0.5, 0.1190), (0.4, 0.1200),
            (0.3, 0.1080), (0.2, 0.0870), (0.1, 0.0600), (0.05, 0.0440),
            (0.025, 0.0320), (0.0, 0.0), (0.025, -0.0080), (0.05, -0.0090),
            (0.1, -0.0080), (0.2, -0.0045), (0.3, -0.0010), (0.4, 0.0010),
            (0.5, 0.0020), (0.6, 0.0020), (0.7, 0.0015), (0.8, 0.0008),
            (0.9, 0.0003), (0.95, 0.0001), (1.0, -0.0005)
        ]
    },
    "Eppler E205": {
        "name": "Eppler E205",
        "type": "Cambered",
        "camber_description": "3% camber, low-drag design",
        "max_thickness": 0.105,
        "max_camber": 0.03,
        "zero_lift_angle": -2.5,
        "stall_angle": 15,
        "base_drag": 0.006,
        "description": "A low-drag laminar flow airfoil designed by Richard Eppler. Optimized for sailplanes and efficient cruise flight. Has lower thickness and refined pressure distribution.",
        "coordinates": [
            (1.0, 0.0010), (0.95, 0.0100), (0.9, 0.0185), (0.8, 0.0330),
            (0.7, 0.0448), (0.6, 0.0536), (0.5, 0.0588), (0.4, 0.0596),
            (0.3, 0.0545), (0.2, 0.0430), (0.1, 0.0290), (0.05, 0.0215),
            (0.025, 0.0158), (0.0, 0.0), (0.025, -0.0108), (0.05, -0.0145),
            (0.1, -0.0185), (0.2, -0.0220), (0.3, -0.0225), (0.4, -0.0208),
            (0.5, -0.0178), (0.6, -0.0138), (0.7, -0.0098), (0.8, -0.0060),
            (0.9, -0.0028), (0.95, -0.0015), (1.0, -0.0010)
        ]
    }
}
QUIZ_QUESTIONS = [
    {
        "question": "What happens to lift when velocity doubles (all else constant)?",
        "options": ["Lift doubles", "Lift quadruples", "Lift stays the same", "Lift halves"],
        "correct": 1,
        "explanation": "Lift is proportional to V². When velocity doubles, lift increases by a factor of 4 (quadruples)."
    },
    {
        "question": "Which force acts opposite to the direction of aircraft motion?",
        "options": ["Lift", "Weight", "Drag", "Thrust"],
        "correct": 2,
        "explanation": "Drag is the aerodynamic force that opposes the motion of the aircraft through the air."
    },
    {
        "question": "What is the angle of attack?",
        "options": [
            "The angle between the wing and the fuselage",
            "The angle between the chord line and the oncoming airflow",
            "The angle between the wing and the ground",
            "The angle between lift and drag vectors"
        ],
        "correct": 1,
        "explanation": "Angle of attack is the angle between the airfoil's chord line and the direction of the oncoming airflow (relative wind)."
    },
    {
        "question": "What happens when the angle of attack exceeds the stall angle?",
        "options": [
            "Lift increases rapidly",
            "Drag decreases",
            "Lift decreases and drag increases sharply",
            "The aircraft speeds up"
        ],
        "correct": 2,
        "explanation": "Beyond the stall angle, airflow separates from the upper surface, causing a dramatic loss of lift and sharp increase in drag."
    },
    {
        "question": "Why do cambered airfoils generate lift even at zero angle of attack?",
        "options": [
            "Because they are heavier on the bottom",
            "Because their curved shape creates a pressure difference",
            "Because they spin in the air",
            "Because gravity pulls them up"
        ],
        "correct": 1,
        "explanation": "The asymmetric (cambered) shape forces air to travel faster over the top surface, creating lower pressure above and higher pressure below, generating lift even at zero angle of attack."
    }
]
THEORY_CONCEPTS = [
    {
        "title": "Airfoil",
        "icon": "✈️",
        "short": "The cross-sectional shape of a wing designed to produce aerodynamic force.",
        "detail": """An airfoil is the cross-sectional profile of a wing, blade, or sail. It is specifically shaped to generate a useful aerodynamic force (lift) when moving through air.
Key features include:
• **Leading edge**: The front of the airfoil
• **Trailing edge**: The back of the airfoil  
• **Upper surface**: Also called the suction side
• **Lower surface**: Also called the pressure side
• **Chord line**: Straight line from leading to trailing edge"""
    },
    {
        "title": "Lift",
        "icon": "⬆️",
        "short": "The upward force generated by pressure difference between upper and lower surfaces.",
        "detail": """Lift is the aerodynamic force that acts perpendicular to the direction of airflow. It is primarily generated by the pressure difference between the upper and lower surfaces.
**The Lift Equation:**
L = 0.5 × ρ × V² × S × CL
Where:
• ρ (rho) = air density (kg/m³)
• V = velocity (m/s)
• S = wing area (m²)
• CL = lift coefficient
Lift increases with the **square of velocity** — doubling speed quadruples lift!"""
    },
    {
        "title": "Drag",
        "icon": "⬅️",
        "short": "The backward force opposing the motion of an object through air.",
        "detail": """Drag is the aerodynamic force that opposes motion through air. Types include:
• **Parasitic drag**: From friction and shape
• **Induced drag**: Byproduct of lift (wingtip vortices)
• **Wave drag**: At transonic/supersonic speeds
**The Drag Equation:**
D = 0.5 × ρ × V² × S × CD
Drag also increases with velocity squared."""
    },
    {
        "title": "Angle of Attack (α)",
        "icon": "📐",
        "short": "The angle between the chord line and the direction of oncoming airflow.",
        "detail": """The angle of attack (α) is one of the most important parameters in aerodynamics.
As angle of attack increases:
• Lift increases (up to stall angle)
• Drag also increases
• Pressure difference grows
The relationship is approximately linear until stall:
CL ≈ 0.1 × (α - α₀) per degree
Where α₀ is the zero-lift angle."""
    },
    {
        "title": "Stall",
        "icon": "🚨",
        "short": "Loss of lift when the angle of attack exceeds the critical angle.",
        "detail": """Stall occurs when the angle of attack exceeds the critical (stall) angle:
• Airflow separates from upper surface
• Dramatic decrease in lift
• Sharp increase in drag
• Loss of control if not recovered
**Important**: Stall does NOT mean engine failure — it's an aerodynamic condition!
Recovery: Reduce angle of attack and increase airspeed."""
    },
    {
        "title": "Lift Coefficient (CL)",
        "icon": "📈",
        "short": "A dimensionless number representing the lifting ability of an airfoil.",
        "detail": """The lift coefficient characterizes the lift-generating capability of an airfoil.
CL depends on:
• Angle of attack (primary factor)
• Airfoil shape (camber, thickness)
• Reynolds number
For thin airfoils: CL ≈ 2π × α (radians)
A higher CL means more lift per unit of dynamic pressure and wing area."""
    },
    {
        "title": "Drag Coefficient (CD)",
        "icon": "📉",
        "short": "A dimensionless number representing the drag produced by an airfoil.",
        "detail": """The drag coefficient quantifies resistance in fluid flow.
CD has components:
• CD₀: Zero-lift drag (parasitic)
• Induced drag: k × CL²
**Total drag coefficient:**
CD = CD₀ + k × CL²
Lower CD = more aerodynamically efficient shape."""
    },
    {
        "title": "Lift-to-Drag Ratio (L/D)",
        "icon": "⚖️",
        "short": "The ratio of lift to drag, indicating aerodynamic efficiency.",
        "detail": """L/D ratio is the primary measure of aerodynamic efficiency.
**Typical L/D values:**
• Sailplanes: 30-60
• Commercial aircraft: 15-20
• Fighter jets: 5-10
• Flat plate: ~1
Maximum L/D determines the best glide ratio — how far you can glide per unit altitude lost."""
    },
    {
        "title": "Camber",
        "icon": "🌊",
        "short": "The asymmetry of the airfoil shape — the curvature of the mean line.",
        "detail": """Camber is the curvature of the mean camber line (halfway between upper and lower surfaces).
**Types:**
• **Zero camber (symmetric)**: No lift at α=0
• **Positive camber**: Generates lift at α=0
• **Reflexed camber**: Trailing edge curves up
More camber = more lift at low angles, but also more drag and earlier stall."""
    },
    {
        "title": "Chord Line",
        "icon": "📏",
        "short": "The straight line from the leading edge to the trailing edge.",
        "detail": """The chord line is the reference line for measuring:
• Angle of attack
• Camber (deviation from chord)
• Wing area (chord × span)
**Chord length** is the distance from leading to trailing edge."""
    },
    {
        "title": "Pressure Difference",
        "icon": "🔄",
        "short": "The pressure variation between surfaces that creates lift.",
        "detail": """The fundamental mechanism of lift:
• **Upper surface**: Faster air → Lower pressure
• **Lower surface**: Slower air → Higher pressure
This pressure difference creates a net upward force.
**Bernoulli's principle** (simplified):
Higher velocity = Lower pressure"""
    },
    {
        "title": "Flow Separation",
        "icon": "💨",
        "short": "When airflow detaches from the surface, leading to stall.",
        "detail": """Flow separation occurs when the boundary layer cannot follow the airfoil contour.
**Causes:**
• Angle of attack too high
• Abrupt surface changes
• Very low Reynolds number
**Effects:**
• Turbulent wake forms
• Upper surface pressure increases
• Lift drops, drag increases
• Airfoil stalls"""
    }
]
# ============================================
# UTILITY FUNCTIONS
# ============================================
def parse_airfoil_text(text, name="Custom Airfoil"):
    """Parse airfoil coordinate text and return coordinates and geometry."""
    lines = text.strip().split('\n')
    coords = []
    parsed_name = name
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Try to parse as coordinates
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
            except ValueError:
                if i == 0:
                    parsed_name = line
        elif i == 0:
            parsed_name = line
    
    if len(coords) < 4:
        return None
    
    # Normalize coordinates
    x_vals = [c[0] for c in coords]
    x_min, x_max = min(x_vals), max(x_vals)
    x_range = x_max - x_min if x_max != x_min else 1
    
    normalized = [(( c[0] - x_min) / x_range, c[1] / x_range) for c in coords]
    
    # Find leading edge (minimum x)
    le_idx = min(range(len(normalized)), key=lambda i: normalized[i][0])
    
    # Split into upper and lower
    upper = normalized[:le_idx+1]
    lower = normalized[le_idx:]
    
    upper = sorted(upper, key=lambda p: p[0])
    lower = sorted(lower, key=lambda p: p[0])
    
    # Calculate geometry
    def interpolate(surface, x_val):
        if not surface:
            return 0
        if x_val <= surface[0][0]:
            return surface[0][1]
        if x_val >= surface[-1][0]:
            return surface[-1][1]
        
        for i in range(len(surface) - 1):
            if surface[i][0] <= x_val <= surface[i+1][0]:
                t = (x_val - surface[i][0]) / (surface[i+1][0] - surface[i][0] + 1e-10)
                return surface[i][1] + t * (surface[i+1][1] - surface[i][1])
        return 0
    
    stations = np.linspace(0, 1, 21)
    max_thickness = 0
    max_thickness_loc = 0
    max_camber = 0
    max_camber_loc = 0
    
    for x_st in stations:
        y_up = interpolate(upper, x_st)
        y_low = interpolate(lower, x_st)
        thickness = y_up - y_low
        camber = (y_up + y_low) / 2
        
        if thickness > max_thickness:
            max_thickness = thickness
            max_thickness_loc = x_st
        if abs(camber) > abs(max_camber):
            max_camber = camber
            max_camber_loc = x_st
    
    # Estimate airfoil type
    if abs(max_camber) < 0.005:
        airfoil_type = "Symmetric"
        stall_angle = 15
    elif abs(max_camber) < 0.03:
        airfoil_type = "Mildly cambered"
        stall_angle = 16
    else:
        airfoil_type = "Highly cambered"
        stall_angle = 13
    
    return {
        "name": parsed_name,
        "coordinates": normalized,
        "max_thickness": max_thickness,
        "max_thickness_loc": max_thickness_loc,
        "max_camber": abs(max_camber),
        "max_camber_loc": max_camber_loc,
        "type": airfoil_type,
        "stall_angle": stall_angle,
        "zero_lift_angle": -abs(max_camber) * 80,
        "base_drag": 0.012 if abs(max_camber) > 0.03 else 0.008
    }
def calculate_aerodynamics(airfoil, angle_of_attack, velocity, air_density, chord_length, wing_span):
    """Calculate aerodynamic forces and coefficients."""
    S = chord_length * wing_span
    dynamic_pressure = 0.5 * air_density * velocity ** 2
    
    zero_lift_angle = airfoil.get("zero_lift_angle", 0)
    stall_angle = airfoil.get("stall_angle", 15)
    cl_alpha = 0.1  # per degree
    
    # Calculate CL
    CL = cl_alpha * (angle_of_attack - zero_lift_angle)
    stall_warning = False
    
    if angle_of_attack > stall_angle:
        stall_warning = True
        excess = angle_of_attack - stall_angle
        cl_max = cl_alpha * (stall_angle - zero_lift_angle)
        CL = cl_max * np.exp(-0.15 * excess ** 2)
    
    if angle_of_attack < -5:
        CL = cl_alpha * (-5 - zero_lift_angle)
    
    # Calculate CD
    base_drag = airfoil.get("base_drag", 0.01)
    max_thickness = airfoil.get("max_thickness", 0.12)
    induced_drag = 0.01 * CL ** 2
    thickness_drag = max_thickness * 0.05
    CD = base_drag + induced_drag + thickness_drag
    
    if angle_of_attack > 12:
        CD += 0.002 * (angle_of_attack - 12) ** 2
    if angle_of_attack > stall_angle:
        CD += 0.005 * (angle_of_attack - stall_angle) ** 2
    
    CD = max(CD, 0.001)
    
    # Calculate forces
    lift = dynamic_pressure * S * CL
    drag = dynamic_pressure * S * CD
    ld_ratio = CL / CD if CD != 0 else 0
    
    # Generate explanation
    if stall_warning:
        explanation = f"⚠️ **STALL WARNING**: At {angle_of_attack}° angle of attack (beyond stall angle of {stall_angle}°), airflow separates from the upper surface. This causes dramatic lift loss and drag increase. This is a dangerous condition!"
    elif angle_of_attack < 0:
        explanation = f"At {angle_of_attack}° (negative angle), the airfoil produces {'reduced lift due to camber' if airfoil.get('max_camber', 0) > 0 else 'negative lift (downforce)'}. Drag remains low."
    elif angle_of_attack == 0:
        explanation = f"At 0° angle of attack, {'the cambered shape still generates lift through pressure difference' if airfoil.get('max_camber', 0) > 0 else 'this symmetric airfoil generates zero lift'}."
    elif angle_of_attack <= 5:
        explanation = f"At {angle_of_attack}°, the airfoil deflects air downward creating a pressure difference. Upper surface has low pressure, lower surface has high pressure. This is an efficient operating condition."
    elif angle_of_attack <= 12:
        explanation = f"At {angle_of_attack}°, significant lift is produced. Drag is increasing but L/D ratio is {'good' if ld_ratio > 10 else 'moderate'}. Airflow is mostly attached."
    else:
        explanation = f"At {angle_of_attack}°, near the stall angle of {stall_angle}°. Lift is near maximum but drag is high. Small increases could cause stall!"
    
    return {
        "CL": round(CL, 4),
        "CD": round(CD, 4),
        "lift": round(lift, 2),
        "drag": round(drag, 2),
        "ld_ratio": round(ld_ratio, 2),
        "stall_warning": stall_warning,
        "explanation": explanation
    }
def generate_curve_data(airfoil, velocity, air_density, chord_length, wing_span):
    """Generate CL and CD curves over angle of attack range."""
    aoa_range = np.arange(-5, 20.5, 0.5)
    data = []
    
    for aoa in aoa_range:
        result = calculate_aerodynamics(airfoil, aoa, velocity, air_density, chord_length, wing_span)
        data.append({
            "aoa": aoa,
            "CL": result["CL"],
            "CD": result["CD"],
            "ld_ratio": result["ld_ratio"]
        })
    
    return data
def plot_airfoil_visualization(airfoil, angle_of_attack, stall_warning, CL, CD):
    """Create airfoil visualization with flow arrows."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get coordinates
    coords = airfoil.get("coordinates", [])
    if not coords:
        ax.text(0.5, 0.5, "No coordinate data available", ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Transform coordinates based on angle of attack
    angle_rad = np.radians(-angle_of_attack)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_coords = np.array([c[0] - 0.5 for c in coords])
    y_coords = np.array([c[1] for c in coords])
    
    x_rot = x_coords * cos_a - y_coords * sin_a + 0.5
    y_rot = x_coords * sin_a + y_coords * cos_a + 0.5
    
    # Set background
    ax.set_facecolor('#f0f9ff')
    
    # Draw airflow arrows
    if stall_warning:
        # Separated flow
        for y_off in [-0.15, -0.05, 0.05, 0.15]:
            ax.annotate('', xy=(0.25, 0.5 + y_off), xytext=(0.0, 0.5 + y_off),
                       arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=1.5, alpha=0.5))
        # Turbulent flow on top
        for i, y_off in enumerate([-0.05, 0, 0.05]):
            ax.plot([0.6, 0.7, 0.8, 0.9], 
                   [0.5 + y_off + 0.15, 0.5 + y_off + 0.2 + 0.02*np.sin(i), 
                    0.5 + y_off + 0.15, 0.5 + y_off + 0.2],
                   'r-', lw=1.5, alpha=0.6)
    else:
        # Smooth airflow
        for y_off in [-0.2, -0.1, 0, 0.1, 0.2]:
            ax.annotate('', xy=(0.2, 0.5 + y_off), xytext=(-0.05, 0.5 + y_off),
                       arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=1.5, alpha=0.4))
            ax.annotate('', xy=(1.1, 0.5 + y_off * 0.8), xytext=(0.85, 0.5 + y_off * 0.9),
                       arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=1.5, alpha=0.4))
    
    # Draw airfoil
    airfoil_color = '#fecaca' if stall_warning else '#bfdbfe'
    edge_color = '#dc2626' if stall_warning else '#1e40af'
    ax.fill(x_rot, y_rot, color=airfoil_color, edgecolor=edge_color, linewidth=2, alpha=0.9)
    
    # Draw chord line
    chord_x = [0.5 - 0.5 * cos_a, 0.5 + 0.5 * cos_a]
    chord_y = [0.5 - 0.5 * sin_a, 0.5 + 0.5 * sin_a]
    ax.plot(chord_x, chord_y, '--', color='#64748b', lw=1, alpha=0.5)
    
     # Draw lift arrow (direction flips when CL is negative)
    if abs(CL) > 0.01:
        lift_len = min(abs(CL) * 0.15, 0.3) * np.sign(CL)
        ax.annotate('', xy=(0.5, 0.5 + lift_len), xytext=(0.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color='#16a34a', lw=3))
        ax.text(0.52, 0.5 + lift_len, 'Lift', fontsize=10, fontweight='bold', color='#16a34a')
    
    # Draw drag arrow (points to the right, opposing forward motion)
    drag_len = min(CD * 3, 0.25)
    ax.annotate('', xy=(0.5 + drag_len, 0.5), xytext=(0.5, 0.5),
               arrowprops=dict(arrowstyle='->', color='#dc2626', lw=3))
    ax.text(0.5 + drag_len + 0.02, 0.52, 'Drag', fontsize=10, fontweight='bold', color='#dc2626', ha='left')
    
    # Pressure labels (swap when lift is negative)
    if CL >= 0:
        ax.text(0.5, 0.75, 'Low Pressure ↑', ha='center', fontsize=9, color='#2563eb', fontweight='bold', alpha=0.7)
        ax.text(0.5, 0.25, 'High Pressure ↓', ha='center', fontsize=9, color='#dc2626', fontweight='bold', alpha=0.7)
    else:
        ax.text(0.5, 0.75, 'High Pressure ↓', ha='center', fontsize=9, color='#dc2626', fontweight='bold', alpha=0.7)
        ax.text(0.5, 0.25, 'Low Pressure ↑', ha='center', fontsize=9, color='#2563eb', fontweight='bold', alpha=0.7)
    
    # Angle of attack arc
    arc_angles = np.linspace(0, -angle_of_attack, 20)
    arc_r = 0.15
    arc_x = 0.5 + arc_r * np.cos(np.radians(arc_angles))
    arc_y = 0.5 + arc_r * np.sin(np.radians(arc_angles))
    ax.plot(arc_x, arc_y, '-', color='#f59e0b', lw=2)
    ax.text(0.68, 0.5 + 0.02 * (-1 if angle_of_attack >= 0 else 1), f'α = {angle_of_attack}°', 
            fontsize=10, color='#f59e0b', fontweight='bold')
    
    # Stall warning box
    if stall_warning:
        ax.text(0.85, 0.9, '⚠️ STALL', fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='#dc2626', alpha=0.9),
               ha='center', va='center')
    
    # Info label
    ax.text(0.02, 0.98, f"{airfoil.get('name', 'Airfoil')} | AoA: {angle_of_attack}°", 
            fontsize=9, color='#475569', transform=ax.transAxes, va='top')
    
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(0.1, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig
def plot_cl_vs_aoa(curve_data, current_aoa, stall_angle):
    """Plot CL vs Angle of Attack."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    aoa = [d["aoa"] for d in curve_data]
    cl = [d["CL"] for d in curve_data]
    
    ax.plot(aoa, cl, 'g-', lw=2, label='CL')
    ax.axvline(x=current_aoa, color='#f59e0b', linestyle='--', lw=1.5, label=f'Current (α={current_aoa}°)')
    ax.axvline(x=stall_angle, color='#dc2626', linestyle=':', lw=1.5, label=f'Stall (α={stall_angle}°)')
    
    ax.set_xlabel('Angle of Attack (°)', fontsize=10)
    ax.set_ylabel('Lift Coefficient (CL)', fontsize=10)
    ax.set_title('CL vs Angle of Attack', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    return fig
def plot_cd_vs_aoa(curve_data, current_aoa):
    """Plot CD vs Angle of Attack."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    aoa = [d["aoa"] for d in curve_data]
    cd = [d["CD"] for d in curve_data]
    
    ax.plot(aoa, cd, 'r-', lw=2, label='CD')
    ax.axvline(x=current_aoa, color='#f59e0b', linestyle='--', lw=1.5, label=f'Current (α={current_aoa}°)')
    
    ax.set_xlabel('Angle of Attack (°)', fontsize=10)
    ax.set_ylabel('Drag Coefficient (CD)', fontsize=10)
    ax.set_title('CD vs Angle of Attack', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    return fig
def plot_drag_polar(curve_data):
    """Plot CL vs CD (Drag Polar)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    cd = [d["CD"] for d in curve_data]
    cl = [d["CL"] for d in curve_data]
    
    ax.plot(cd, cl, 'purple', lw=2)
    
    ax.set_xlabel('Drag Coefficient (CD)', fontsize=10)
    ax.set_ylabel('Lift Coefficient (CL)', fontsize=10)
    ax.set_title('Drag Polar: CL vs CD', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    return fig
def plot_airfoil_shape(airfoil, title="Airfoil Shape"):
    """Plot just the airfoil shape."""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    coords = airfoil.get("coordinates", [])
    if not coords:
        ax.text(0.5, 0.5, "No coordinates", ha='center', va='center')
        return fig
    
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    
    ax.fill(x, y, color='#bfdbfe', edgecolor='#1e40af', linewidth=2, alpha=0.8)
    ax.plot([0, 1], [0, 0], '--', color='#64748b', lw=1, alpha=0.5)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 0.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_facecolor('#f8fafc')
    ax.axis('off')
    
    plt.tight_layout()
    return fig
# ============================================
# SESSION STATE
# ============================================
if 'custom_airfoils' not in st.session_state:
    st.session_state.custom_airfoils = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = [None] * len(QUIZ_QUESTIONS)
# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🎛️ Simulator", "📐 Custom Airfoil", "📊 Compare Airfoils", 
     "📚 Learn Theory", "🧠 Quiz", "📋 About Project"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Airfoil Library")
st.sidebar.markdown(f"**Predefined:** {len(PREDEFINED_AIRFOILS)}")
st.sidebar.markdown(f"**Custom:** {len(st.session_state.custom_airfoils)}")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.8rem; color: #94a3b8;'>
⚠️ <b>Educational Tool</b><br>
Simplified approximations for visualization. Not for real aircraft design.
</div>
""", unsafe_allow_html=True)
# ============================================
# PAGE: HOME
# ============================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">✈️ Interactive Airfoil Lift & Drag Visualizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn how airfoil shape, velocity, and angle of attack affect lift and drag</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>🎓 Interdisciplinary Engineering Project</b><br>
    Combining <b>Aeronautical Engineering</b> and <b>Computer Science</b> to create an interactive 
    educational platform for understanding aerodynamic forces.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Simulation", use_container_width=True, type="primary"):
            st.session_state.page = "simulator"
            st.rerun()
    with col2:
        if st.button("📐 Add Custom Airfoil", use_container_width=True):
            st.session_state.page = "custom"
            st.rerun()
    
    st.markdown("### 🎯 What You Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎛️ Interactive Simulator**
        
        Adjust angle of attack, velocity, air density, and wing dimensions. See real-time lift and drag calculations.
        """)
    
    with col2:
        st.markdown("""
        **📊 Compare Airfoils**
        
        Compare two airfoil profiles side-by-side. Analyze lift, drag, and efficiency differences.
        """)
    
    with col3:
        st.markdown("""
        **📐 Custom Airfoils**
        
        Paste airfoil coordinate data. Parse geometry and use in simulations.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Interactive Graphs**
        
        View CL vs AoA, CD vs AoA, and drag polar curves that update dynamically.
        """)
    
    with col2:
        st.markdown("""
        **📚 Learn Theory**
        
        Understand lift, drag, stall, camber, angle of attack, and more with simple explanations.
        """)
    
    with col3:
        st.markdown("""
        **🧠 Test Knowledge**
        
        Take a quiz on aerodynamic fundamentals and test your understanding.
        """)
    
    st.markdown("### 📚 Predefined Airfoil Library")
    
    cols = st.columns(6)
    for i, name in enumerate(PREDEFINED_AIRFOILS.keys()):
        with cols[i]:
            st.markdown(f"**{name}**")
    
    st.markdown("""
    <div class="disclaimer">
    <b>⚠️ Educational Tool Disclaimer:</b> This tool uses simplified aerodynamic approximations for educational visualization. 
    Results are not CFD-accurate and should not be used for real aircraft design without validation.
    </div>
    """, unsafe_allow_html=True)
# ============================================
# PAGE: SIMULATOR
# ============================================
elif page == "🎛️ Simulator":
    st.markdown("## 🎛️ Airfoil Simulator")
    st.markdown("Adjust parameters and observe how lift and drag change in real time")
    
    # Get all available airfoils
    all_airfoils = {**PREDEFINED_AIRFOILS, **st.session_state.custom_airfoils}
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Input Parameters")
        
        selected_name = st.selectbox("Airfoil", list(all_airfoils.keys()))
        airfoil = all_airfoils[selected_name]
        
        st.caption(f"{airfoil.get('type', 'Unknown')} • {airfoil.get('camber_description', '')}")
        
        angle_of_attack = st.slider("Angle of Attack (°)", -5.0, 20.0, 5.0, 0.5)
        velocity = st.slider("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0)
        air_density = st.slider("Air Density (kg/m³)", 0.5, 1.5, 1.225, 0.025)
        chord_length = st.slider("Chord Length (m)", 0.1, 2.0, 1.0, 0.05)
        wing_span = st.slider("Wing Span (m)", 0.5, 10.0, 5.0, 0.5)
        
        wing_area = chord_length * wing_span
        st.info(f"**Wing Area (S)** = {chord_length} × {wing_span} = **{wing_area:.2f} m²**")
    
    with col2:
        # Calculate results
        result = calculate_aerodynamics(airfoil, angle_of_attack, velocity, air_density, chord_length, wing_span)
        
        # Airfoil visualization
        fig = plot_airfoil_visualization(airfoil, angle_of_attack, result["stall_warning"], result["CL"], result["CD"])
        st.pyplot(fig)
        plt.close()
        
        # Results cards
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Lift Coefficient (CL)", f"{result['CL']:.4f}")
            st.metric("Lift Force", f"{result['lift']:.2f} N")
        with col_b:
            st.metric("Drag Coefficient (CD)", f"{result['CD']:.4f}")
            st.metric("Drag Force", f"{result['drag']:.2f} N")
        with col_c:
            st.metric("L/D Ratio", f"{result['ld_ratio']:.2f}")
            if result["stall_warning"]:
                st.error("⚠️ STALL")
            else:
                st.success("✅ Normal")
    
    # Explanation
    if result["stall_warning"]:
        st.markdown(f"""<div class="stall-warning">{result['explanation']}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="info-box">💡 <b>What is happening?</b><br>{result['explanation']}</div>""", unsafe_allow_html=True)
    
    # Graphs
    st.markdown("### 📈 Performance Curves")
    curve_data = generate_curve_data(airfoil, velocity, air_density, chord_length, wing_span)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_cl_vs_aoa(curve_data, angle_of_attack, airfoil.get("stall_angle", 15))
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        fig2 = plot_cd_vs_aoa(curve_data, angle_of_attack)
        st.pyplot(fig2)
        plt.close()
    
    # Drag polar
    fig3 = plot_drag_polar(curve_data)
    st.pyplot(fig3)
    plt.close()
    
    # Equations
    st.markdown("### 📐 Equations Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Lift Force:**
        ```
        L = 0.5 × ρ × V² × S × CL
        L = 0.5 × {air_density} × {velocity}² × {wing_area:.2f} × {result['CL']:.4f}
        L = {result['lift']:.2f} N
        ```
        """)
    with col2:
        st.markdown(f"""
        **Drag Force:**
        ```
        D = 0.5 × ρ × V² × S × CD
        D = 0.5 × {air_density} × {velocity}² × {wing_area:.2f} × {result['CD']:.4f}
        D = {result['drag']:.2f} N
        ```
        """)
    
    st.markdown("""
    <div class="disclaimer">
    ⚠️ This tool uses simplified aerodynamic approximations for educational visualization. Results are not CFD-accurate.
    </div>
    """, unsafe_allow_html=True)
# ============================================
# PAGE: CUSTOM AIRFOIL
# ============================================
elif page == "📐 Custom Airfoil":
    st.markdown("## 📐 Custom Airfoil")
    st.markdown("Add custom airfoils by pasting coordinates or copying from predefined airfoils")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Method")
        
        input_method = st.radio("Choose input method:", ["📋 Paste Coordinates", "📂 Copy from Predefined"])
        
        if input_method == "📂 Copy from Predefined":
            selected = st.selectbox("Select airfoil to copy:", list(PREDEFINED_AIRFOILS.keys()))
            if st.button("Load Coordinates"):
                af = PREDEFINED_AIRFOILS[selected]
                coord_text = f"{af['name']}\n"
                coord_text += "\n".join([f"{c[0]:.4f} {c[1]:.4f}" for c in af["coordinates"]])
                st.session_state.coord_text = coord_text
        
        airfoil_name = st.text_input("Airfoil Name", "My Custom Airfoil")
        
        coord_text = st.text_area(
            "Coordinate Data (x y per line)",
            value=st.session_state.get('coord_text', ''),
            height=300,
            placeholder="""NACA 2412
1.0000 0.0013
0.9500 0.0074
0.9000 0.0126
...
0.0000 0.0000
..."""
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            parse_btn = st.button("🔍 Parse Airfoil", type="primary", use_container_width=True)
        with col_b:
            if st.button("Load Sample", use_container_width=True):
                sample = """NACA 2412
1.0000 0.0013
0.9500 0.0074
0.9000 0.0126
0.8000 0.0214
0.7000 0.0286
0.6000 0.0347
0.5000 0.0398
0.4000 0.0433
0.3000 0.0435
0.2000 0.0375
0.1000 0.0256
0.0000 0.0000
0.1000 -0.0121
0.2000 -0.0180
0.3000 -0.0201
0.4000 -0.0190
0.5000 -0.0160
0.6000 -0.0120
0.7000 -0.0082
0.8000 -0.0050
0.9000 -0.0025
1.0000 -0.0013"""
                st.session_state.coord_text = sample
                st.rerun()
    
    with col2:
        st.markdown("### Analysis Results")
        
        if parse_btn and coord_text:
            parsed = parse_airfoil_text(coord_text, airfoil_name)
            
            if parsed:
                st.session_state.parsed_airfoil = parsed
                st.success(f"✅ Successfully parsed **{parsed['name']}** with {len(parsed['coordinates'])} points")
            else:
                st.error("❌ Could not parse airfoil data. Check format: each line should have 'x y' coordinates.")
        
        if 'parsed_airfoil' in st.session_state:
            parsed = st.session_state.parsed_airfoil
            
            # Show shape
            fig = plot_airfoil_shape(parsed, f"Shape: {parsed['name']}")
            st.pyplot(fig)
            plt.close()
            
            # Geometry analysis
            st.markdown("#### 📊 Geometry Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Name:** {parsed['name']}")
                st.markdown(f"**Type:** {parsed['type']}")
                st.markdown(f"**Max Thickness:** {parsed['max_thickness']*100:.1f}%")
                st.markdown(f"**Thickness Location:** {parsed['max_thickness_loc']*100:.0f}% chord")
            with col_b:
                st.markdown(f"**Max Camber:** {parsed['max_camber']*100:.2f}%")
                st.markdown(f"**Camber Location:** {parsed['max_camber_loc']*100:.0f}% chord")
                st.markdown(f"**Est. Stall Angle:** {parsed['stall_angle']}°")
                st.markdown(f"**Coordinate Points:** {len(parsed['coordinates'])}")
            
            # Add to library
            if st.button("➕ Add to Airfoil Library", type="primary", use_container_width=True):
                st.session_state.custom_airfoils[parsed['name']] = parsed
                st.success(f"✅ **{parsed['name']}** added to library! Use it in Simulator and Compare pages.")
        else:
            st.info("👈 Paste coordinate data and click **Parse Airfoil** to see analysis")
    
    # Show custom library
    if st.session_state.custom_airfoils:
        st.markdown("---")
        st.markdown("### 📚 Your Custom Airfoils")
        for name, af in st.session_state.custom_airfoils.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{name}** - {af['type']} • {af['max_thickness']*100:.1f}% thick • {af['max_camber']*100:.2f}% camber")
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{name}"):
                    del st.session_state.custom_airfoils[name]
                    st.rerun()
    
    # Format guide
    st.markdown("---")
    st.markdown("""
    ### 📖 Coordinate Format Guide
    - First line can be the airfoil name (optional)
    - Each subsequent line: `x_coordinate  y_coordinate`
    - Coordinates can be space, tab, or comma separated
    - Upper surface first (trailing edge → leading edge), then lower surface
    - x values normalized 0 to 1 (or will be auto-normalized)
    - You can copy coordinates from airfoil databases like UIUC Airfoil Database
    """)
# ============================================
# PAGE: COMPARE AIRFOILS
# ============================================
elif page == "📊 Compare Airfoils":
    st.markdown("## 📊 Compare Airfoils")
    st.markdown("Compare two airfoils side-by-side under the same conditions")
    
    all_airfoils = {**PREDEFINED_AIRFOILS, **st.session_state.custom_airfoils}
    airfoil_names = list(all_airfoils.keys())
    
    if len(airfoil_names) < 2:
        st.warning("Need at least 2 airfoils to compare. Add custom airfoils or use predefined ones.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            airfoil_a_name = st.selectbox("🔵 Airfoil A", airfoil_names, index=0)
        with col2:
            airfoil_b_name = st.selectbox("🟠 Airfoil B", airfoil_names, index=min(1, len(airfoil_names)-1))
        with col3:
            angle_of_attack = st.slider("Angle of Attack (°)", -5.0, 20.0, 8.0, 0.5)
        with col4:
            velocity = st.slider("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            air_density = st.slider("Air Density (kg/m³)", 0.5, 1.5, 1.225, 0.025)
        with col2:
            chord_length = st.slider("Chord Length (m)", 0.1, 2.0, 1.0, 0.05)
        with col3:
            wing_span = st.slider("Wing Span (m)", 0.5, 10.0, 5.0, 0.5)
        
        airfoil_a = all_airfoils[airfoil_a_name]
        airfoil_b = all_airfoils[airfoil_b_name]
        
        result_a = calculate_aerodynamics(airfoil_a, angle_of_attack, velocity, air_density, chord_length, wing_span)
        result_b = calculate_aerodynamics(airfoil_b, angle_of_attack, velocity, air_density, chord_length, wing_span)
        
        # Shape comparison
        st.markdown("### Airfoil Shapes")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🔵 {airfoil_a_name}**")
            fig_a = plot_airfoil_shape(airfoil_a)
            st.pyplot(fig_a)
            plt.close()
            st.caption(f"{airfoil_a.get('type', '')} • {airfoil_a.get('camber_description', '')}")
        
        with col2:
            st.markdown(f"**🟠 {airfoil_b_name}**")
            fig_b = plot_airfoil_shape(airfoil_b)
            st.pyplot(fig_b)
            plt.close()
            st.caption(f"{airfoil_b.get('type', '')} • {airfoil_b.get('camber_description', '')}")
        
        # Results comparison
        st.markdown("### Performance Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🔵 {airfoil_a_name} @ {angle_of_attack}°**")
            st.metric("CL", f"{result_a['CL']:.4f}")
            st.metric("CD", f"{result_a['CD']:.4f}")
            st.metric("L/D Ratio", f"{result_a['ld_ratio']:.2f}")
            st.metric("Lift", f"{result_a['lift']:.1f} N")
            st.metric("Drag", f"{result_a['drag']:.1f} N")
            if result_a['stall_warning']:
                st.error("⚠️ STALL")
            else:
                st.success("✅ Normal")
        
        with col2:
            st.markdown(f"**🟠 {airfoil_b_name} @ {angle_of_attack}°**")
            st.metric("CL", f"{result_b['CL']:.4f}")
            st.metric("CD", f"{result_b['CD']:.4f}")
            st.metric("L/D Ratio", f"{result_b['ld_ratio']:.2f}")
            st.metric("Lift", f"{result_b['lift']:.1f} N")
            st.metric("Drag", f"{result_b['drag']:.1f} N")
            if result_b['stall_warning']:
                st.error("⚠️ STALL")
            else:
                st.success("✅ Normal")
        
        # Geometry comparison table
        st.markdown("### Geometry Comparison")
        comparison_data = {
            "Property": ["Type", "Max Thickness", "Max Camber", "Stall Angle", "Zero-Lift Angle"],
            f"🔵 {airfoil_a_name}": [
                airfoil_a.get("type", "N/A"),
                f"{airfoil_a.get('max_thickness', 0)*100:.1f}%",
                f"{airfoil_a.get('max_camber', 0)*100:.1f}%",
                f"{airfoil_a.get('stall_angle', 15)}°",
                f"{airfoil_a.get('zero_lift_angle', 0)}°"
            ],
            f"🟠 {airfoil_b_name}": [
                airfoil_b.get("type", "N/A"),
                f"{airfoil_b.get('max_thickness', 0)*100:.1f}%",
                f"{airfoil_b.get('max_camber', 0)*100:.1f}%",
                f"{airfoil_b.get('stall_angle', 15)}°",
                f"{airfoil_b.get('zero_lift_angle', 0)}°"
            ]
        }
        st.table(comparison_data)
        
        # Bar charts
        st.markdown("### Coefficient Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Coefficients
        x = np.arange(3)
        width = 0.35
        vals_a = [result_a['CL'], result_a['CD'], result_a['ld_ratio']/10]
        vals_b = [result_b['CL'], result_b['CD'], result_b['ld_ratio']/10]
        
        axes[0].bar(x - width/2, vals_a, width, label=airfoil_a_name, color='#3b82f6')
        axes[0].bar(x + width/2, vals_b, width, label=airfoil_b_name, color='#f97316')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['CL', 'CD', 'L/D ÷10'])
        axes[0].set_title('Coefficient Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Forces
        x = np.arange(2)
        vals_a = [result_a['lift'], result_a['drag']]
        vals_b = [result_b['lift'], result_b['drag']]
        
        axes[1].bar(x - width/2, vals_a, width, label=airfoil_a_name, color='#3b82f6')
        axes[1].bar(x + width/2, vals_b, width, label=airfoil_b_name, color='#f97316')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Lift (N)', 'Drag (N)'])
        axes[1].set_title('Force Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Curve comparisons
        st.markdown("### CL vs AoA Comparison")
        curve_a = generate_curve_data(airfoil_a, velocity, air_density, chord_length, wing_span)
        curve_b = generate_curve_data(airfoil_b, velocity, air_density, chord_length, wing_span)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        aoa = [d["aoa"] for d in curve_a]
        cl_a = [d["CL"] for d in curve_a]
        cl_b = [d["CL"] for d in curve_b]
        cd_a = [d["CD"] for d in curve_a]
        cd_b = [d["CD"] for d in curve_b]
        
        axes[0].plot(aoa, cl_a, 'b-', lw=2, label=airfoil_a_name)
        axes[0].plot(aoa, cl_b, color='#f97316', lw=2, label=airfoil_b_name)
        axes[0].axvline(x=angle_of_attack, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Angle of Attack (°)')
        axes[0].set_ylabel('CL')
        axes[0].set_title('CL vs AoA')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(aoa, cd_a, 'b-', lw=2, label=airfoil_a_name)
        axes[1].plot(aoa, cd_b, color='#f97316', lw=2, label=airfoil_b_name)
        axes[1].axvline(x=angle_of_attack, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Angle of Attack (°)')
        axes[1].set_ylabel('CD')
        axes[1].set_title('CD vs AoA')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
# ============================================
# PAGE: LEARN THEORY
# ============================================
elif page == "📚 Learn Theory":
    st.markdown("## 📚 Learn Aerodynamic Theory")
    st.markdown("Student-friendly explanations of fundamental aerodynamic concepts")
    
    for concept in THEORY_CONCEPTS:
        with st.expander(f"{concept['icon']} **{concept['title']}** — {concept['short']}", expanded=False):
            st.markdown(concept['detail'])
    
    st.markdown("""
    <div class="info-box">
    <b>🎯 Ready to Apply Your Knowledge?</b><br>
    Head to the <b>Simulator</b> to see these concepts in action, or take the <b>Quiz</b> to test your understanding!
    </div>
    """, unsafe_allow_html=True)
# ============================================
# PAGE: QUIZ
# ============================================
elif page == "🧠 Quiz":
    st.markdown("## 🧠 Aerodynamics Quiz")
    st.markdown("Test your understanding of lift, drag, angle of attack, stall, and airfoils")
    
    # Results display
    if st.session_state.quiz_submitted:
        score = sum(1 for i, ans in enumerate(st.session_state.quiz_answers) 
                   if ans == QUIZ_QUESTIONS[i]["correct"])
        percentage = (score / len(QUIZ_QUESTIONS)) * 100
        
        if percentage == 100:
            st.success(f"🏆 **Perfect Score: {score}/{len(QUIZ_QUESTIONS)} ({percentage:.0f}%)**")
            st.balloons()
        elif percentage >= 80:
            st.success(f"🎉 **Great Job: {score}/{len(QUIZ_QUESTIONS)} ({percentage:.0f}%)**")
        elif percentage >= 50:
            st.warning(f"📖 **Good Effort: {score}/{len(QUIZ_QUESTIONS)} ({percentage:.0f}%)** — Review the theory section!")
        else:
            st.error(f"💪 **Keep Studying: {score}/{len(QUIZ_QUESTIONS)} ({percentage:.0f}%)** — The theory section will help!")
        
        if st.button("🔄 Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = [None] * len(QUIZ_QUESTIONS)
            st.rerun()
    
    # Questions
    for i, q in enumerate(QUIZ_QUESTIONS):
        st.markdown(f"### Question {i+1}")
        st.markdown(f"**{q['question']}**")
        
        answer = st.radio(
            f"Select answer for Q{i+1}:",
            options=list(range(len(q["options"]))),
            format_func=lambda x, q=q: f"{chr(65+x)}. {q['options'][x]}",
            key=f"q_{i}",
            index=st.session_state.quiz_answers[i] if st.session_state.quiz_answers[i] is not None else 0,
            disabled=st.session_state.quiz_submitted
        )
        st.session_state.quiz_answers[i] = answer
        
        # Show results if submitted
        if st.session_state.quiz_submitted:
            if answer == q["correct"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Incorrect. The correct answer is **{chr(65+q['correct'])}**. {q['explanation']}")
        
        st.markdown("---")
    
    # Submit button
    if not st.session_state.quiz_submitted:
        if st.button("📝 Submit Answers", type="primary", use_container_width=True):
            if None in st.session_state.quiz_answers:
                st.warning("Please answer all questions before submitting.")
            else:
                st.session_state.quiz_submitted = True
                st.rerun()
# ============================================
# PAGE: ABOUT
# ============================================
elif page == "📋 About Project":
    st.markdown("## 📋 About This Project")
    st.markdown("Interactive Airfoil Lift & Drag Visualizer — An Interdisciplinary Engineering Project")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0a1628 0%, #1a3a6e 100%); color: white; padding: 2rem; border-radius: 12px; margin: 1rem 0;'>
    <h3 style='margin:0; color: white;'>Interactive Airfoil Lift & Drag Visualizer</h3>
    <p style='color: #7dd3fc; margin: 0.5rem 0;'>Aeronautical Engineering × Computer Science</p>
    <p style='color: #bae6fd; margin: 0.5rem 0; font-size: 0.9rem;'>
    A comprehensive tool that enables students to interactively explore how airfoil geometry, 
    velocity, and angle of attack affect aerodynamic forces.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Statement
    st.markdown("### 🔍 Problem Statement")
    st.markdown("""
    Students often find it difficult to understand lift and drag using only formulas and static diagrams. 
    Traditional teaching methods rely on textbooks and theoretical equations that don't provide intuitive 
    understanding of how aerodynamic forces change with different parameters.
    
    **This project provides an interactive visual platform** to understand how airfoil geometry, velocity, 
    and angle of attack affect aerodynamic forces. By allowing real-time parameter adjustment and 
    visual feedback, students can build deeper intuition about aerodynamic principles.
    """)
    
    # Objectives
    st.markdown("### 🎯 Objectives")
    st.markdown("""
    - To visualize lift and drag generation on airfoils interactively
    - To allow users to change aerodynamic parameters and see real-time results
    - To support both predefined and custom airfoil coordinate files
    - To calculate lift, drag, CL, CD, and lift-to-drag ratio using standard equations
    - To help students understand stall conditions and pressure difference
    - To compare different airfoil shapes and their performance characteristics
    - To provide an educational quiz and theory section for comprehensive learning
    """)
    
    # Methodology
    st.markdown("### ⚙️ Methodology")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. Airfoil Data Collection**
        - Predefined airfoils with stored geometry and properties
        
        **2. Custom Airfoil Parsing**
        - Parser for standard coordinate format
        - Automatic geometry estimation
        
        **3. Aerodynamic Calculations**
        - Simplified educational models for CL, CD
        - Force calculations using standard equations
        """)
    with col2:
        st.markdown("""
        **4. Interactive Visualization**
        - Real-time airfoil rendering with Matplotlib
        - Force arrows and pressure labels
        - Stall warnings and dynamic graphs
        
        **5. Comparison & Analysis**
        - Side-by-side airfoil comparison
        - Bar charts and overlay curves
        """)
    
    # Interdisciplinary Connection
    st.markdown("### 🔗 Interdisciplinary Connection")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **✈️ Aeronautical Engineering**
        - Airfoil theory and design principles
        - Lift and drag equations
        - Angle of attack and stall behavior
        - Pressure distribution concepts
        - Performance comparison methods
        """)
    with col2:
        st.success("""
        **💻 Computer Science**
        - Python/Streamlit application development
        - Interactive data visualization
        - Real-time computation
        - File parsing and data processing
        - User interface design
        """)
    
    # Applications
    st.markdown("### 🚀 Applications")
    st.markdown("""
    - Engineering education and classroom demonstrations
    - Student self-study and concept reinforcement
    - Preliminary airfoil selection for design projects
    - Understanding trade-offs between different airfoil shapes
    - Quiz-based assessment of aerodynamic knowledge
    """)
    
    # Advantages
    st.markdown("### ✅ Advantages")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - ✓ Interactive real-time parameter adjustment
        - ✓ Visual airfoil rendering with force arrows
        - ✓ Support for custom airfoil coordinates
        - ✓ Side-by-side comparison mode
        - ✓ Student-friendly explanations
        """)
    with col2:
        st.markdown("""
        - ✓ Stall detection and warning system
        - ✓ Multiple interactive graphs
        - ✓ Built-in theory and quiz sections
        - ✓ No installation required (Streamlit)
        - ✓ Professional presentation quality
        """)
    
    # Limitations
    st.markdown("### ⚠️ Limitations")
    st.warning("""
    - Uses simplified aerodynamic approximations — not CFD-accurate
    - Does not account for Reynolds number effects in detail
    - 3D wing effects are simplified
    - No compressibility effects (Mach number not considered)
    - Stall modeling is approximate
    - Results should not be used for real aircraft design
    """)
    
    # Future Scope
    st.markdown("### 🔮 Future Scope")
    st.markdown("""
    - 🔬 Add real CFD visualization using computational methods
    - 🛠️ Integrate XFOIL for accurate airfoil analysis
    - 💾 Add database for saving simulations and results
    - 👤 Add user login and personalized dashboards
    - 📄 Export results as PDF reports
    - 🌬️ Add wind tunnel data comparison
    - 🧊 Add 3D wing visualization
    - 🤖 AI-assisted airfoil design optimization
    """)
    
    st.markdown("""
    <div class="disclaimer">
    <b>⚠️ Educational Tool Disclaimer:</b> This tool uses simplified aerodynamic approximations for educational visualization. 
    Results are not CFD-accurate and should not be used for real aircraft design without validation.
    </div>
    """, unsafe_allow_html=True)
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 1rem;'>
    ✈️ <b>Interactive Airfoil Lift & Drag Visualizer</b><br>
    An interdisciplinary project combining Aeronautical Engineering and Computer Science<br>
    <small>⚠️ Educational visualization tool — simplified approximations, not for real aircraft design</small>
</div>
""", unsafe_allow_html=True)
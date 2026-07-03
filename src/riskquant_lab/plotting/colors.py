# # src/riskquant_lab/plotting/colors.py
# """Color palettes for RiskQuant solutions."""

# from enum import StrEnum, unique


# @unique
# class BrandColors(StrEnum):
#     """Core brand colors for RiskQuant solutions."""

#     # Core Brand
#     DARK = "#181A1F"  # Charcoal
#     PRIMARY = "#E30076"  # Magenta
#     SECONDARY = "#6C63FF"  # Electric Violet
#     ACCENT = "#00C2D1"  # Cyan

#     # Neutrals
#     WHITE = "#FFFFFF"
#     LIGHT = "#F8F9FA"
#     GRAY_100 = "#F1F3F5"
#     GRAY_300 = "#DEE2E6"
#     GRAY_500 = "#ADB5BD"
#     GRAY_700 = "#495057"
#     GRAY_900 = "#212529"

#     # Accent
#     SECONDARY = "#5B2EFF"  # Example violet
#     ACCENT = "#00C2A8"  # Example teal

#     # Status
#     SUCCESS = "#16A34A"
#     WARNING = "#F59E0B"
#     ERROR = "#DC2626"
#     INFO = "#2563EB"

#     # Backgrounds
#     BACKGROUND = "#FFFFFF"
#     BACKGROUND_DARK = DARK
#     SURFACE = "#FFFFFF"
#     SURFACE_DARK = "#24272D"

#     # Text
#     TEXT_PRIMARY = DARK
#     TEXT_SECONDARY = "#6C757D"
#     TEXT_INVERSE = WHITE

#     # Borders
#     BORDER = "#DEE2E6"
#     BORDER_DARK = "#3A3F47"


# @unique
# class RiskGradient(StrEnum):
#     """Escalating monochromatic gradient for VaR levels."""

#     LOW = "#FFA3CB"  # Soft pink (e.g., 90% VaR)
#     MEDIUM = "#F74C98"  # Vibrant pink (e.g., 95% VaR)
#     HIGH = "#E30076"  # Brand Magenta (e.g., 99% VaR)
#     EXTREME = "#8A0048"  # Deep plum (e.g., 99.9% VaR)


# @unique
# class DarkModeAccents(StrEnum):
#     """High-contrast neon accents for dark mode / terminal aesthetics."""

#     CYAN = "#00E5FF"
#     GREEN = "#00FF66"
#     YELLOW = "#FFEA00"
#     RED = "#FF1744"

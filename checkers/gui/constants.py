from __future__ import annotations

# Board appearance
BOARD_BG_LIGHT = "#EEEED2"   # light square
BOARD_BG_DARK  = "#769656"   # dark/playable square
BOARD_HL_SQ    = "#F6F669"   # selected square highlight
BOARD_HL_DEST  = "#9AE66E"   # destination highlight
BOARD_LASTMOVE = "#F1C40F"   # last move highlight path

# Piece appearance
PIECE_BLACK_FILL = "#222222"
PIECE_BLACK_OUTLINE = "#FFFFFF"
PIECE_RED_FILL   = "#C0392B"
PIECE_RED_OUTLINE = "#FFFFFF"
KING_TEXT_COLOR = "#FFD700"  # gold-ish

# Layout
SQUARE_SIZE = 72   # pixels per square (window will be ~8*SQUARE_SIZE)
BOARD_SIZE = 8 * SQUARE_SIZE

# UI Fonts
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_NORMAL = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_BUTTON = ("Segoe UI", 10)

# Colors for text and UI elements
TEXT_COLOR_NORMAL = "#000000"
TEXT_COLOR_DISABLED = "#888888"
HINT_COLOR = "#AA66FF"

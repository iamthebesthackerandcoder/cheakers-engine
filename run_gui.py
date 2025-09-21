from __future__ import annotations

from config import setup_logging, ensure_dirs
from checkers.gui.tk_app import CheckersUI


def main() -> None:
    setup_logging()
    ensure_dirs()
    app = CheckersUI()
    app.mainloop()


if __name__ == "__main__":
    main()

"""PyInstaller entry point for the desktop app."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from secondbrain.desktop import main

    main()

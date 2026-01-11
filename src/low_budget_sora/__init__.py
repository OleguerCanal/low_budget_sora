import os

DIR = os.path.dirname(os.path.abspath(__file__))
COMIC_SANS_MS_FONT_PATH = os.path.join(DIR, "data", "Comic Sans MS.ttf")
CONFIG_FILE_PATH = os.path.join(DIR, "training", "config.yaml")
CHECKPOINTS_DIR = os.path.join(DIR, "checkpoints")
DEBUG_GIF_PATH = os.path.join(DIR, "debug.gif")
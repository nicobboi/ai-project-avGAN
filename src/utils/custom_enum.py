from enum import IntEnum

class FPS(IntEnum):
    FPS_30 = int(1000 / 30)
    FPS_60 = int(1000 / 60)
    FPS_120 = int(1000 / 120)

class SampleWindowSize(IntEnum):
    WS_1024 = 1024
    WS_2048 = 2048
    WS_4096 = 4096
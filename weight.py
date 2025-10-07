from gpiozero import DigitalInputDevice, DigitalOutputDevice
from collections import deque
import time
import os

# Pin HX711 sesuai wiring
dout = DigitalInputDevice(5)     # DOUT HX711
pd_sck = DigitalOutputDevice(6)  # SCK HX711

# Buffer untuk moving average
buf = deque(maxlen=20)

# Nilai default
OFFSET = 0
SCALE = 1  

CAL_FILE = "calibration.dat"

# ===== Fungsi dasar HX711 =====
def is_ready():
    return dout.value == 0

def read_raw():
    while not is_ready():
        time.sleep(0.0001)
    data = 0
    for _ in range(24):
        pd_sck.on()
        data = (data << 1) | dout.value
        pd_sck.off()
    # Gain pulse
    pd_sck.on(); pd_sck.off()
    # Konversi signed 24-bit
    if data & 0x800000:
        data |= ~0xffffff
    return data

def read_average(times=10):
    total = 0
    for _ in range(times):
        total += read_raw()
    return total / times

# ===== Simpan & Load Kalibrasi =====
def save_calibration(offset, scale):
    with open(CAL_FILE, "w") as f:
        f.write(f"{offset},{scale}\n")

def load_calibration():
    global OFFSET, SCALE
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE, "r") as f:
            line = f.readline().strip()
            try:
                OFFSET, SCALE = map(float, line.split(","))
                #print(f"Kalibrasi ditemukan: OFFSET={OFFSET:.2f}, SCALE={SCALE:.4f}")
                return True
            except:
                print("File kalibrasi rusak, kalibrasi ulang dibutuhkan.")
    return False

# ===== Tare =====
def tare():
    global OFFSET
    print("Tare... kosongkan timbangan")
    time.sleep(2)
    OFFSET = read_average(20)
    print(f"Offset diset: {OFFSET:.2f}")

# ===== Kalibrasi =====
def calibrate(known_weight_kg):
    global SCALE
    print(f"Letakkan beban {known_weight_kg} kg untuk kalibrasi...")
    time.sleep(5)
    raw_val = read_average(20)
    SCALE = (raw_val - OFFSET) / known_weight_kg
    print(f"SCALE diset: {SCALE:.4f} counts per kg")
    save_calibration(OFFSET, SCALE)

# ===== Moving Average Filter =====
def get_weight_filtered():
    raw_val = read_average(5)
    weight = (raw_val - OFFSET) / SCALE
    return weight

# ===== Program utama =====
if not load_calibration():
    tare()  
    calibrate(known_weight_kg=1.45)  # kalibrasi sekali


def get_weight():
	return get_weight_filtered()
	

	

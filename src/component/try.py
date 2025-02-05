import numpy as np
import matplotlib.pyplot as plt

def adjust_angle(angle):
    if angle < -45:
        return -(90 + angle)
    else:
        return -angle

angles = np.linspace(-90, 0, 10)  # Generating angles from -90° to 0°
adjusted_angles = [adjust_angle(a) for a in angles]

plt.plot(angles, adjusted_angles, marker='o', label="Adjusted Angles")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(-45, color='red', linestyle='--', label="Threshold (-45°)")
plt.xlabel("Original Angle")
plt.ylabel("Adjusted Angle")
plt.legend()
plt.title("Angle Adjustment Function")
plt.show()

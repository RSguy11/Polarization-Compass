#INTENSITY-> S1, S2, DOLP, AOLP - MEASURED
 
import cv2
import numpy as np
import polanalyser as pa
import matplotlib.pyplot as plt
 
# Read image and demosaicing
# Path to raw intensity
img_raw = cv2.imread("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/new_underwater_test/2026-02-24_12-05-44_burst001_frame010.png", 0)
 
#Uses Polanalyser to demosaic
img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
 
def normalize_for_display(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)
 
# Calculate the Stokes Vectors
image_list = [img_000, img_045, img_090, img_135]
angles = np.deg2rad([0, 45, 90, 135])
img_stokes = pa.calcStokes(image_list, angles)
 
# Decompose the Stokes vector into its components
img_s0, img_s1, img_s2 = cv2.split(img_stokes)
 
# Convert the Stokes vector to Intensity, DoLP and AoLP
img_intensity = pa.cvtStokesToIntensity(img_stokes)
img_dolp = pa.cvtStokesToDoLP(img_stokes)
img_aolp = pa.cvtStokesToAoLP(img_stokes)
 
print("aolp min:", np.min(img_aolp), "max:", np.max(img_aolp))
 
img_aolp = np.rad2deg(img_aolp)
 
print("aolp min:", np.min(img_aolp), "max:", np.max(img_aolp))
plt.show()
 
#This is to verify Visualization of all-sky polarization images referenced in the instrument, scattering, and solar principal planes
 
def compute_angle_for_mueller(rows=2048, cols=2448, center_x=1224, center_y=1024):
    y_coord, x_coord = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dx = x_coord - center_x
    dy = y_coord - center_y
    angle_matrix = np.degrees(np.arctan2(dy, dx))
    angle_matrix = (angle_matrix + 360) % 360
    angle_matrix = angle_matrix % 180
    return angle_matrix
 
# Create the angle matrix - precomputed
beta1 = compute_angle_for_mueller()
# beta2 = np.fliplr(compute_angle_for_mueller())
beta2 = compute_angle_for_mueller()
betaspp = 2 * beta2
cosbeta = np.cos(np.radians(betaspp))
sinbeta = np.sin(np.radians(betaspp))
negcosbeta = -cosbeta
 
DOLP = np.sqrt(img_s1**2 + img_s2**2) / img_s0
 
# Calculate AolpSPP
AolpIPP = img_aolp
temp1 = np.deg2rad(AolpIPP+beta2)
aolpspp2 = 0.5 * np.rad2deg(np.arctan2(np.sin(2*temp1),np.cos(2*temp1)))#///////////////////////////////////////////////////////
 
# --------- Display Stokes images (Colab-safe) ---------
 
plt.figure(figsize=(15,10))
 
plt.subplot(2,3,1)
plt.imshow(normalize_for_display(img_s0), cmap='gray')
plt.title("S0")
plt.axis('off')
 
plt.subplot(2,3,2)
plt.imshow(normalize_for_display(img_s1), cmap='gray')
plt.title("S1")
plt.axis('off')
 
plt.subplot(2,3,3)
plt.imshow(normalize_for_display(img_s2), cmap='gray')
plt.title("S2")
plt.axis('off')
 
plt.subplot(2,3,4)
plt.imshow(normalize_for_display(img_dolp), cmap='jet')
plt.title("DoLP")
plt.axis('off')
plt.colorbar()
 
plt.subplot(2,3,5)
plt.imshow(normalize_for_display(img_aolp), cmap='jet')
plt.title("S1")
plt.axis('off')
plt.title('AoLP IPP')
 
plt.subplot(2,3,6)
plt.imshow(normalize_for_display(aolpspp2), cmap='jet')
plt.title("S1")
plt.axis('off')
plt.title('AoLP SPP')
 
# plt.figure(figsize=(8, 8))
# plt.imshow(beta1, cmap='jet')
# plt.title('β2 Angle Matrix (Degrees), matrix representation')
# plt.colorbar(label='Angle (°)')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
 
#The images are flipped to show the top-down view, with east appearing on the right
# flipped_Aolp = np.fliplr(aolpspp2)
flipped_Aolp = np.fliplr(aolpspp2)
 
 
# plt.figure(1)
# plt.imshow(flipped_Aolp, cmap='jet')
# plt.title('Flipped AoLP SPP, aerial view')
# plt.colorbar()
# plt.axis('equal')
# plt.show()
 
# flipped_Dolp = np.fliplr(DOLP)
# plt.figure(2)
# plt.imshow(flipped_Dolp, cmap='jet')
# plt.title('flipped DOLP')
# plt.colorbar()
# plt.axis('equal')
# plt.show()
 
# aolp_meas = aolpspp2
aolp_meas_full = flipped_Aolp
dolp_meas = DOLP
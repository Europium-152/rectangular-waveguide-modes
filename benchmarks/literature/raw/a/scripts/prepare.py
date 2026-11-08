"""Derived page images used by fig4.py, fig5*.py and doane*.py (L1a).
Run in the scratch folder after the pdfimages commands listed in ../README.md."""
import math, sys
sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import load_gray, ink, edges_v, edges_h, deskew

img = load_gray('al_native-000.png'); I = ink(img)
# Alessandri Fig. 4: deskew about the axis origin by the mean tilt of the left and bottom axes
for (vx0, vx1, vy0, vy1, hy0, hy1, hx0, hx1, out) in [
        (1695, 1725, 360, 880, 878, 915, 1730, 2470, 'al_p2_deskew.png'),     # Fig. 4 (0.6296 deg)
        (1668, 1700, 1260, 1770, 1775, 1810, 1700, 2420, 'al_p2_deskew5.png')]:  # Fig. 5 (0.5397 deg)
    pv, tv, _ = edges_v(I, vx0, vx1, vy0, vy1); ph, th, _ = edges_h(I, hy0, hy1, hx0, hx1)
    ang = (math.degrees(math.atan(pv[0])) + math.degrees(math.atan(-ph[0]))) / 2
    y0 = (ph[0] * pv[1] + ph[1]) / (1 - ph[0] * pv[0]); x0 = pv[0] * y0 + pv[1]
    deskew(img, -ang, (x0, y0)).save(out)
    print(out, 'rotation %.4f deg about (%.1f, %.1f)' % (ang, x0, y0))

# Doane & Anderson Fig. 9: crop of the page photograph (offset (880, 1800) is used by doane.py)
load_gray('dn7-000.png').crop((880, 1800, 2263, 3000)).save('dn_f9.png')

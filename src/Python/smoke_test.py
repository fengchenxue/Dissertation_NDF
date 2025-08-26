# -*- coding: utf-8 -*-
import sys, math, random
import faulthandler; faulthandler.enable()

# Make sure Python can find the built .pyd
sys.path.append(r"E:\Code\Dissertation\Dissertation_NDF\build\Release")

import ndf_py
print("Imported ndf_py OK")

N = 64
Dx = [1.0 / N] * N
Dy = [1.0 / N] * N
ndf = ndf_py.PiecewiseLinearNDF(Dx, Dy)
print("PiecewiseLinearNDF created")

def norm(x, y, z):
    L = (x*x + y*y + z*z) ** 0.5 or 1.0
    return ndf_py.Vec3f(x / L, y / L, z / L)

def hemi():
    # return a unit vector on the upper hemisphere (z > 0)
    while True:
        x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)
        v = norm(x, y, z)
        if v.z > 1e-8:
            return v

w  = ndf_py.Vec3f(0.0, 0.0, 1.0)
wh = ndf_py.Vec3f(0.0, 0.0, 1.0)
print("G1(normal, normal) =", ndf.G1(w, wh))

vals = [ndf.G1(hemi(), hemi()) for _ in range(2000)]
print("min =", min(vals), "max =", max(vals), "mean =", sum(vals) / len(vals))
print("SMOKE TEST PASSED")

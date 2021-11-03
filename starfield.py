# reference ==> http://www.shadertoy.com/view/XtBSWt

import taichi as ti
import handy_shader_functions as hsf

ti.init(arch=ti.gpu)

res_x = 512
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))
bandPass = 720.
angleDisp = 6.28 / (bandPass+1.0)
particlesCount = 200
particleLifetime = 50.
particleMaxSize = 357.5
polarRadiusClip = 0.05
polarRadiusMax = 0.75
polarRadiusDelta = polarRadiusMax - polarRadiusClip
timeDelta = bandPass


@ti.func
def rand(x):
    return hsf.fract(ti.sin(x*78.) * 1e4)


@ti.func
def polar(P):
    return ti.Vector([P.norm(), ti.atan2(P.y, P.x)])


@ti.func
def cart(P):
    return ti.Vector([P.x * ti.cos(P.y), P.x * ti.sin(P.y)])


@ti.kernel
def render(t: ti.f32):
    # draw something on your canvas
    for i, j in pixels:
        R = ti.Vector([res_x, res_y])
        P = ti.Vector([0.0,0.0])
        frag = ti.Vector([i+0.5,j+0.5]) - .5 * R
        fragPolar = polar(frag)

        lenCenter = (R * 0.5).norm()
        c = 0.0
        globTime = t / particleLifetime
        a = 0.0
        for b in range(particlesCount):
            localTime = globTime + timeDelta * (2. * a - 1.) + a
            particleTime = hsf.fract(localTime)
            spaceTransform = particleTime ** 8
            P.x = lenCenter * (polarRadiusClip + polarRadiusDelta * a + spaceTransform)
            if (abs(P.x - fragPolar.x) > particleMaxSize):
                continue
            P.y = ti.floor(particleTime + bandPass * rand(ti.floor(localTime))) * angleDisp
            c += particleMaxSize * spaceTransform * hsf.clamp(1. - (cart(P) - frag).norm(), 0., 1.)
            a += 1.0 / particlesCount
        color = ti.Vector([c, c, c])
        pixels[i, j] = color


gui = ti.GUI("Canvas", res=(res_x, res_y))

for i in range(100000):
    t = i * 0.3
    render(t)
    gui.set_image(pixels)
    gui.show()

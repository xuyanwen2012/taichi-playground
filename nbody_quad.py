# Heavily inspired by 'N-body example', 'Quadtree example', and 'Tree Gravity example'.
#
# https://github.com/taichi-dev/taichi/blob/master/examples/tree_gravity.py#L18
#
import taichi as ti
import math
import numpy as np

# , print_preprocessed=True
ti.init(arch=ti.gpu)

# Program related
RES = 512
DIM = 2

# N-body related
# NUM_MAX_PARTICLE = 1600
NUM_MAX_PARTICLE = 8192
num_particles = ti.field(ti.i32, shape=())

particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_table = ti.root.dense(ti.i, NUM_MAX_PARTICLE)

# particle_table.place(particle_pos, particle_vel)
particle_table.place(particle_pos).place(particle_vel)


# N-body physics related
R0 = 0.05
DT = 1e-5
STEPS = 160
EPS = 1e-3
G = -1e1

# Quadtree related
K = 2
T_MAX_DEPTH = 7
T_NODES = K ** T_MAX_DEPTH


@ti.kernel
def initialize():
    for i in range(num_particles[None]):
        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        particle_pos[i] = 0.5 + ti.Vector([ti.cos(a), ti.sin(a)]) * r


@ti.kernel
def substep():
    for i in range(num_particles[None]):
        acc = ti.Vector([0.0, 0.0])

        p = particle_pos[i]
        for j in range(num_particles[None]):
            if i != j:
                r = p - particle_pos[j]
                x = R0 / r.norm(1e-4)
                # Molecular force: https://www.zhihu.com/question/38966526
                acc += EPS * (x ** 13 - x ** 7) * r
                # Long-distance gravity force:
                acc += G * (x ** 3) * r

        particle_vel[i] += acc * DT

    for i in range(num_particles[None]):
        particle_pos[i] += particle_vel[i] * DT


#
# @ti.kernel
# def code_test():
#     # Permutation
#     # [0, 0]
#     # [0, 1]
#     # [1, 0]
#     # [1, 1]
#     for I in ti.grouped(ti.ndrange(*([2] * DIM))):
#         print(I)


num_particles[None] = 1600
gui = ti.GUI('N-body Star')

initialize()
while gui.running and not gui.get_event(ti.GUI.ESCAPE):
    gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
    gui.show()
    for i in range(STEPS):
        substep()

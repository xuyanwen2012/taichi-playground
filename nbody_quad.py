""" Heavily inspired by 'N-body example', 'Quadtree example', and
'Tree Gravity example'.

https://github.com/taichi-dev/taichi/blob/master/examples/tree_gravity.py#L18

"""
import taichi as ti
import math

ti.init(arch=ti.gpu)
if not hasattr(ti, 'jkl'):
    ti.jkl = ti.indices(1, 2, 3)

# N-body related
DIM = 2
NUM_MAX_PARTICLE = 8192  # 2^13

# Using this table to store all the information (pos, vel, mass) of particles
# Currently using SoA memory model
particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_mass = ti.field(dtype=ti.f32)
particle_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
num_particles = ti.field(dtype=ti.i32, shape=())

# # For some reason, this structure has ~28fps (vs ~25fps)
# particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32, shape=NUM_MAX_PARTICLE)
# particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32, shape=NUM_MAX_PARTICLE)

# N-body physics related
R0 = 0.05
DT = 1e-5
STEPS = 160
EPS = 1e-3
G = -1e1

# Quadtree related
T_MAX_DEPTH = NUM_MAX_PARTICLE
T_MAX_NODES = 4 * T_MAX_DEPTH  # T_MAX_NODES = K ** T_MAX_DEPTH
LEAF = -1
TREE = -2

# Each node contains information about the node mass, the centroid position,
# and the particle which it contains in ID
node_mass = ti.field(ti.f32)
node_particle_id = ti.field(ti.i32)
node_centroid_pos = ti.Vector.field(DIM, ti.f32)

# Node table contains information
node_children = ti.field(ti.i32)
node_table = ti.root.dense(ti.i, T_MAX_NODES)
node_table.place(node_mass, node_particle_id, node_centroid_pos)  # AoS here
node_table.dense(indices={2: ti.jk, 3: ti.jkl}[DIM], dimensions=2).place(
    node_children)  # ????
node_table_len = ti.field(dtype=ti.i32, shape=())


@ti.func
def alloc_node():
    """
    Increment the current node table length, clear and set initial values
    (mass/centroid) to zeros of the allocated. The children information is
    stored in the 'node_children' table.
    :return: the ID of the just allocated node
    """
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < T_MAX_NODES

    node_mass[ret] = 0
    node_centroid_pos[ret] = particle_pos[0] * 0

    # indicate the 4 children to be LEAF as well
    node_particle_id[ret] = LEAF
    for which in ti.grouped(ti.ndrange(*([2] * DIM))):
        node_children[ret, which] = LEAF
    return ret


@ti.func
def alloc_particle():
    """
    Always use this function to obtain an new particle id to operate on.
    :return: The ID of the just allocated particle
    """
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret


@ti.func
def alloc_a_node_for_particle(particle_id, parent, parent_geo_center,
                              parent_geo_size):
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]

    depth = 0
    while depth < T_MAX_DEPTH:
        already_particle_id = node_particle_id[parent]
        if already_particle_id == LEAF:
            break
        if already_particle_id != TREE:
            node_particle_id[parent] = TREE
            # Subtract pos/mass of the particle from the parent node
            already_pos = particle_pos[already_particle_id]
            already_mass = particle_mass[already_particle_id]
            node_centroid_pos[parent] -= already_pos * already_mass
            node_mass[parent] -= already_mass

        node_centroid_pos[parent] += position * mass
        node_mass[parent] += mass

        which = abs(position > parent_geo_center)
        child = node_children[parent, which]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which] = child
        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

        depth = depth + 1

    node_particle_id[parent] = particle_id
    node_centroid_pos[parent] = position * mass
    node_mass[parent] = mass


@ti.kernel
def build_tree():
    """
    Once the 'particle table' is populated, we can construct a 'node table',
    which contains all the node information, and construct the child table as
    well.
    :return:
    """
    node_table_len[None] = 0
    alloc_node()

    # Making sure not to parallelize this loop, for each particle operation.
    particle_id = 0
    while particle_id < num_particles[None]:
        alloc_a_node_for_particle(particle_id, 0, particle_pos[0] * 0 + 0.5,
                                  1.0)

        particle_id = particle_id + 1


# The O(NlogN) kernel using quadtree
@ti.kernel
def substep():
    for _ in range(num_particles[None]):
        pass


# The O(N^2) kernel algorithm
@ti.kernel
def substep_raw():
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


@ti.kernel
def initialize(num_p: ti.i32):
    """
    Randomly set the initial position of the particles to start with. Note
    set a value to 'num_particles[None]' taichi field to indicate.
    :return: None
    """
    for i in range(num_p):
        particle_id = alloc_particle()

        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        particle_pos[particle_id] = 0.5 + ti.Vector([ti.cos(a), ti.sin(a)]) * r

        # TODO: random initial starting velocity and mass
        # particle_mass[particle_id] = tl.randRange(0.0, 1.5)


if __name__ == '__main__':
    gui = ti.GUI('N-body Star')

    initialize(1600)
    while gui.running and not gui.get_event(ti.GUI.ESCAPE):
        gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        for _ in range(STEPS):
            build_tree()
            substep()
            # substep_raw()

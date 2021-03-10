import taichi as ti

ti.init(arch=ti.gpu)
if not hasattr(ti, 'jkl'):
    ti.jkl = ti.indices(1, 2, 3)

# content data
NUM_MAX_ENTITY = 1024

# position is used for spatial partitioning
entity_pos = ti.Vector.field(2, dtype=ti.f32)

entity_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_ENTITY)
entity_table.place(entity_pos)
entity_table_len = ti.field(dtype=ti.i32, shape=())

# data structure
T_MAX_DEPTH = NUM_MAX_ENTITY
T_MAX_NODES = 4 * T_MAX_DEPTH
LEAF = -1
TREE = -2

node_children = ti.field(ti.i32)

# The fundamental informations each node must have are {entity_id, children}
node_entity_id = ti.field(dtype=ti.i32)
node_table = ti.root.dense(ti.i, T_MAX_NODES).place(node_entity_id)
# define this way so you can access children using
# 'node_children[node_index, which]' to get the child node at index which.
node_table.dense(indices={2: ti.jk, 3: ti.jkl}[2], dimensions=2).place(
    node_children)
node_table_len = ti.field(dtype=ti.i32, shape=())


@ti.func
def alloc_entity():
    ret = ti.atomic_add(entity_table_len[None], 1)
    assert ret < NUM_MAX_ENTITY
    entity_pos[ret] = 0
    return ret


@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < T_MAX_NODES

    # indicate the 4 children to be LEAF as well
    node_entity_id[ret] = LEAF
    # which = {
    #   [0, 0]
    #   [0, 1]
    #   [1, 0]
    #   [1, 1]
    # }
    for which in ti.grouped(ti.ndrange(2)):
        node_children[ret, which] = LEAF
    return ret


@ti.func
def alloc_a_node_for_entity(entity_id, parent, parent_geo_center,
                            parent_geo_size):
    position = entity_pos[entity_id]
    # (Making sure not to parallelize this loop)
    # Traversing down the tree to find a suitable location for the particle.
    depth = 0
    while depth < T_MAX_DEPTH:
        parent_entity_id = node_entity_id[parent]
        if parent_entity_id == LEAF:
            break
        if parent_entity_id != TREE:
            node_entity_id[parent] = TREE

        # Determine which quadrant (as 'child') this particle shout go into.
        which = abs(position > parent_geo_center)
        child = node_children[parent, which]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which] = child

        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent = child
        depth = depth + 1

    node_entity_id[parent] = entity_id


@ti.kernel
def build_tree():
    node_table_len[None] = 0
    alloc_node()
    entity_id = 0
    while entity_id < entity_table_len[None]:
        alloc_a_node_for_entity(entity_id, 0, entity_pos[0] * 0 + 0.5, 1.0)
        entity_id = entity_id + 1


@ti.kernel
def test():
    for which in ti.grouped(ti.ndrange(*([2] * 2))):
        print(which)


if __name__ == '__main__':
    test()

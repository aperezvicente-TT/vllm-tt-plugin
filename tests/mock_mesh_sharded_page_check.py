"""Stage 12 x 14: the per-owner page I/O of the sharded pool on the 2x4 MOCK mesh (no device; readback is not real):
the sharded read (slice into the persistent page + the owner shard) and write (stage + a PER-CHIP page table +
paged_fill_cache) compile and run on a mesh tensor, allocate NOTHING after the warm (GLM53_KV_PAGE_ALLOC_CHECK=1), and
compile no program after the warm (the probe counts DRAM bytes per bank AND program-cache entries before / after every
read and write: the SlotOps copy is one program for every page; the slice fallback, GLM53_KV_PAGE_SLOTOPS=0, must raise).
usage: mock_mesh_sharded_page_check.py <plugin src>"""
import gc, json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["GLM53_KV_PAGE_ALLOC_CHECK"] = "1"
import torch
import ttnn
from vllm_tt_plugin import kv_connector as kc

gc.disable()
C = 8
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=536870912, worker_l1_size=1264512)
def alloc(): return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
NB_LOCAL = 8
# every chip's LOCAL pool: one mesh tensor, per-chip content (ShardTensorToMesh over C x nb_local host blocks)
host = torch.zeros(C * NB_LOCAL, 1, 64, 512, dtype=torch.bfloat16)
pool = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                       mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
assert kc._ttnn_shape(pool) == [NB_LOCAL, 1, 64, 512], kc._ttnn_shape(pool)
io = kc._TtnnMeshPageIO(pool, C)
res = {"nb_local": NB_LOCAL, "C": C}
w0 = alloc()
io.warm()  # creates the persistent read page / stage / table (the pre-capture warm)
res["warm_alloc_bytes_per_bank"] = alloc() - w0
kc.freeze_page_buffers()
base = alloc()
deltas = []
for owner, local in [(0, 1), (3, 5), (7, 7), (5, 2)]:
    b0 = alloc()
    page = io.read_local(owner, local)
    deltas.append(("read", owner, local, alloc() - b0))
    assert tuple(page.shape) == (1, 1, 64, 512), page.shape
for owner, local in [(1, 3), (6, 6), (0, 7), (7, 1)]:
    b0 = alloc()
    io.write_local(owner, local, torch.full((1, 1, 64, 512), 2.0, dtype=torch.bfloat16))
    deltas.append(("write", owner, local, alloc() - b0))
res["per_op_delta_bytes_per_bank"] = deltas
res["delta_after_batch"] = alloc() - base
try:
    io.read_local(8, 0); res["bad_owner"] = "no error (BAD)"
except IndexError as e:
    res["bad_owner"] = "raised: " + str(e)[:60]
other = ttnn.from_torch(torch.zeros(4, 1, 32, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    kc._TtnnMeshPageIO(other, C).read_local(0, 1); res["frozen_new_shape"] = "no error (BAD)"
except RuntimeError as e:
    res["frozen_new_shape"] = "raised: " + str(e)[:80]
res["program_cache_after_warm"] = int(mesh.num_program_cache_entries())
# negative control (review_shard_pool.md B1): the slice fallback keys a program on the page index -> the guard raises
os.environ["GLM53_KV_PAGE_SLOTOPS"] = "0"
kc._SLOT_OPS.clear()
try:
    io.read_local(2, 4); io.read_local(6, 5); res["slice_fallback"] = "no error (BAD)"
except RuntimeError as e:
    res["slice_fallback"] = "raised: " + str(e)[:70]
os.environ["GLM53_KV_PAGE_SLOTOPS"] = "1"
res["pass"] = (res["delta_after_batch"] == 0 and all(d[3] == 0 for d in deltas) and res["bad_owner"].startswith("raised")
               and res["frozen_new_shape"].startswith("raised") and res["slice_fallback"].startswith("raised"))
print("SHARDED_PAGE_ALLOC", json.dumps(res), flush=True)
ttnn.close_mesh_device(mesh)

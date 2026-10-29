"""gx2c3: after the warm (first use creates the persistent buffers, then freeze) page reads / writes allocate nothing.
usage: page_alloc_check2.py <plugin src>"""
import gc, json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["GLM53_KV_PAGE_ALLOC_CHECK"] = "1"
import torch
import ttnn
from vllm_tt_plugin import kv_connector as kc

gc.disable()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=536870912, worker_l1_size=1264512)
def alloc(): return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
cache = ttnn.from_torch(torch.zeros(8, 1, 64, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = {}
w0 = alloc(); kc._ttnn_read_pages(cache, [0]); kc._ttnn_write_pages(cache, [0], torch.zeros(1, 1, 64, 512, dtype=torch.bfloat16))
res["warm_alloc_bytes_per_bank"] = alloc() - w0  # the persistent buffers
kc.freeze_page_buffers()
base = alloc()
pages = kc._ttnn_read_pages(cache, [1, 2, 5])
kc._ttnn_write_pages(cache, [3, 4], torch.ones(2, 1, 64, 512, dtype=torch.bfloat16))
res.update(read_shape=list(pages.shape), delta_after_ops=alloc() - base)
other = ttnn.from_torch(torch.zeros(4, 1, 32, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    kc._ttnn_read_pages(other, [1]); res["frozen_new_shape"] = "no error (BAD)"
except RuntimeError as e:
    res["frozen_new_shape"] = "raised: " + str(e)[:90]
res["pass"] = res["delta_after_ops"] == 0 and res["frozen_new_shape"].startswith("raised")
print("PAGE_ALLOC2", json.dumps(res), flush=True)
ttnn.close_mesh_device(mesh)

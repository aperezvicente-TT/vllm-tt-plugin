"""Mock cluster (host-only): program-cache entries around post-warm page reads / writes must not grow (Stage 12: ttnn.slice keyed on the page index compiled a program per page after the capture). usage: mock_mesh_page_program_check.py <plugin src>; GLM53_KV_PAGE_SLOTOPS=0 = the slice fallback (expected to raise under GLM53_MOCK=1)."""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch, ttnn
from vllm_tt_plugin import kv_connector as K
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
try:
    lat = ttnn.from_torch(torch.zeros(16, 1, 64, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    h = K.TTPagedCacheHandle(lat, (16, 1, 64, 512), torch.bfloat16)
    host = {"l": torch.zeros(16, 1, 64, 512, dtype=torch.bfloat16)}
    K.tt_copy_kv_blocks({"l": h}, host, [0], [0], "d2h"); K.tt_copy_kv_blocks(host, {"l": h}, [0], [0], "h2d")
    getattr(K, "freeze_page_buffers", lambda: None)()
    p0 = mesh.num_program_cache_entries()
    K.tt_copy_kv_blocks({"l": h}, host, [1, 2], [1, 2], "d2h"); p1 = mesh.num_program_cache_entries()
    K.tt_copy_kv_blocks(host, {"l": h}, [5, 6], [5, 6], "h2d"); p2 = mesh.num_program_cache_entries()
    print(f"PC_PROBE {sys.argv[1]}: after warm {p0}, after store d2h [1,2] {p1}, after hit h2d [5,6] {p2}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)

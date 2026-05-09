# %%
import open3d as o3d
from pathlib import Path
path = "~/Projects/NNProject/repopt-3d/third_party/openscene/out/baseline_adapt_test/result_eval_baseline_adapt-test"
# path = "third_party/openscene/out/hpo_fusion_seed3407/ce/final_eval_adapt-test/result_eval_adapter_adapt-test"
# path = "third_party/openscene/out/hpo_fusion_seed3407/ce_h1/final_eval_adapt-test/result_eval_adapter_adapt-test"
path = Path(path)
print(path, path.exists())
p = o3d.io.read_point_cloud(path / "0_input.ply")
o3d.visualization.draw_geometries([p])
# %%
p = o3d.io.read_point_cloud(path / "0_gt.ply")
o3d.visualization.draw_geometries([p])
# %%
p = o3d.io.read_point_cloud(path / "0_fusion.ply")
o3d.visualization.draw_geometries([p])
# %%

# # %%
# # Minimal bar plot (edit the lists below)
# import matplotlib.pyplot as plt  # type: ignore[import-not-found]

# STYLE = "seaborn-v0_8-whitegrid"  # try: "ggplot", "fivethirtyeight", "dark_background"
# X_LABELS = ["Baseline(No Training)", "L_sup", "L_sup + L_graph", "L_sup + L_ent", "L_sup + L_tr"]
# Y_VALUES = [40.04, 48.79, 48.65, 48.89, 47.20]
# # OUT_PATH = "artifacts/barplot.png"

# plt.style.use(STYLE)
# plt.figure(figsize=(10, 5))
# bars = plt.bar(X_LABELS, Y_VALUES, color="#8C1A10", width=0.62)
# ax = plt.gca()
# ax.yaxis.grid(True, linestyle="--", alpha=0.35)
# ax.xaxis.grid(False)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.ylim(0, 60)
# plt.ylabel("mIoU (%)", fontsize=13, fontweight="bold")
# plt.title("mIoU across model configurations", fontsize=20, fontweight="bold", pad=12)
# plt.xticks(fontsize=11, fontweight="bold")
# plt.yticks(fontsize=11, fontweight="bold")

# for b in bars:
#     y = b.get_height()
#     plt.text(
#         b.get_x() + b.get_width() / 2,
#         y + 0.8,
#         f"{y:.2f}%",
#         ha="center",
#         va="bottom",
#         fontsize=12,
#         fontweight="bold",
#     )

# plt.tight_layout()

# ---------------------------------------------------------------------------
#
#   $py \[process\]_5_dataset_eda_plots.py
#       📊 開始分析資料集，總計物件數: 13989
#       🎉 所有圖表已成功生成，儲存路徑為: /home/USER/eda_report_figures
#
# ---------------------------------------------------------------------------

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 繪圖風格設定
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",]                         # 確保跨平台英文字體一致
plt.rcParams["axes.unicode_minus"] = False  # 正常顯示負號
plt.rcParams["axes.labelsize"] = 12         # 軸標籤字型大小
plt.rcParams["axes.titlesize"] = 14         # 標題字型大小
plt.rcParams["xtick.labelsize"] = 10        # 刻度字型大小
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.titlesize"] = 16
sns.set_theme(style="whitegrid")            # 白色網格背景


def clean_percentage_string(val):
    """輔助函式：將 CSV 裡面的 '12.3456%' 字串轉換為浮點數 12.3456"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.replace("%", "")
    return float(val)


def parse_bbox_center_and_size(row):
    """輔助函式：解析 BBox 座標並計算中心點與長寬"""
    try:
        pts = [list(map(int, pt.split(","))) for pt in row["SAMBBox"].split(";")]
        x1, y1 = pts[0][0], pts[0][1]
        x2, y2 = pts[1][0], pts[1][1]

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2
        cy = ymin + h / 2

        # 進行歸一化 (0~1)，這樣不同解析度的照片才能放在一起比較
        norm_cx = cx / row["width"]
        norm_cy = cy / row["height"]
        norm_w = w / row["width"]
        norm_h = h / row["height"]

        return norm_cx, norm_cy, norm_w, norm_h, w, h
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def run_eda_pipeline(csv_path, output_dir):
    # 1. 讀取並清洗資料
    df = pd.read_csv(csv_path)
    df["p-m_value"] = df["p-m_value"].apply(clean_percentage_string)

    # 動態解析幾何特徵
    bbox_features = df.apply(parse_bbox_center_and_size, axis=1)
    df["norm_cx"] = [x[0] for x in bbox_features]
    df["norm_cy"] = [x[1] for x in bbox_features]
    df["norm_w"] = [x[2] for x in bbox_features]
    df["norm_h"] = [x[3] for x in bbox_features]
    df["w"] = [x[4] for x in bbox_features]
    df["h"] = [x[5] for x in bbox_features]
    df["scale"] = np.sqrt(
        df["w"] * df["h"]
    )  # 物件尺度的科學定義：面積的平方根 (Pixels)
    df["aspect_ratio"] = df["w"] / df["h"]  # 長寬比

    # 移除異常值
    df = df.dropna(
        subset=["norm_cx", "norm_cy", "scale", "P_bbox", "R_bbox"]
    ).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📊 開始分析資料集，總計物件數: {len(df)}")

    # =======================================================================
    # 圖一：類別分佈與 SAM 修正行為直方圖 (Label Distribution & Refinement Type)
    # =======================================================================
    plt.figure(figsize=(10, 6))
    # 建立交叉表
    count_matrix = (
        df.groupby(["label", "p-m_name"]).size().unstack(fill_value=0)
    )
    # 排序讓圖表更美觀
    count_matrix["Total"] = count_matrix.get("BNER", 0) + count_matrix.get(
        "TBRR", 0
    )
    count_matrix = count_matrix.sort_values(by="Total", ascending=False).drop(
        columns=["Total"]
    )

    count_matrix.plot(
        kind="bar", stacked=True, color=["#4682B4", "#CD5C5C"], ax=plt.gca()
    )
    plt.title(
        "Class Distribution and Directional Boundary Adjustments via SAM", pad=15
    )
    plt.xlabel("Category (Class)")
    plt.ylabel("Object Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Contraction", "Expansion"], loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "01_label_distribution_behavior.png"), dpi=300
    )
    plt.close()
    
    
#    # 對數圖 #
#    count_matrix.plot(
#        kind="bar", stacked=True, color=["#4682B4", "#CD5C5C"], ax=plt.gca()
#    )
#    
#    # 關鍵修改：將 Y 軸設為對數刻度
#    plt.yscale('log')
#    
#    plt.title(
#        "Class Distribution and Directional Boundary Adjustments via SAM", pad=15
#    )
#    plt.xlabel("Category (Class)")
#    
#    # 建議把 Label 改一下，提醒讀者這是對數圖
#    plt.ylabel("Object Count (Log Scale)")
#    
#    plt.xticks(rotation=45, ha="right")
#    plt.legend(["Contraction", "Expansion"], loc="upper right")
#    plt.tight_layout()
#    plt.savefig(
#        os.path.join(output_dir, "01_label_distribution_behavior_log.png"), dpi=300
#    )
#    plt.close()
    
    # 在柱狀圖頂端「自動標註實際數字」 🔢 #
    ax = plt.gca()
    count_matrix.plot(
        kind="bar", stacked=True, color=["#4682B4", "#CD5C5C"], ax=ax
    )
    
    # 關鍵修改：自動在柱狀圖頂端加上數字
    for container in ax.containers:
        # labels = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in container]
        # 上面那行會分別標註藍色和紅色，如果只想標註「總數」，可以用下面這個進階寫法：
        pass
        
    # 總數標註：
    totals = count_matrix.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + 20, str(int(total)), ha='center', va='bottom', fontsize=8, rotation=90)

    plt.title(
        "Class Distribution and Directional Boundary Adjustments via SAM", pad=15
    )
    plt.xlabel("Category (Class)")
    plt.ylabel("Object Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Contraction", "Expansion"], loc="upper right")
    
    # 因為加了文字，上方留點空間才不會被標題擋到
    plt.ylim(0, max(totals) * 1.15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "01_label_distribution_behavior_number.png"), dpi=300
    )
    plt.close()
     
    
    # ==========================================
    #                   參數
    # ==========================================
    fig_width = 12       # 整張圖的寬度
    fig_height = 6       # 整張圖的高度
    
    bar_width = 0.7      # 直方柱子的寬度 0.7~0.8
    
    text_rotation = 0    # 數字旋轉角度 (0度為橫寫，90度為直寫)
    text_fontsize = 10   # 數字字體大小
    # ==========================================

    plt.figure(figsize=(fig_width, fig_height))
    
    # 建立交叉表
    count_matrix = (
        df.groupby(["label", "p-m_name"]).size().unstack(fill_value=0)
    )
    # 排序讓圖表更美觀
    count_matrix["Total"] = count_matrix.get("BNER", 0) + count_matrix.get(
        "TBRR", 0
    )
    count_matrix = count_matrix.sort_values(by="Total", ascending=False).drop(
        columns=["Total"]
    )

    # 4682b4 SteelBlue, #cd5c5c IndianRed
    ax = plt.gca()
    count_matrix.plot(
        kind="bar", stacked=True, color=["SteelBlue", "#46b4af"], ax=ax, width=bar_width,
    )
    
    # ==========================================
    # 修正後的數字標註（移除不支援的 va 參數）
    # ==========================================
    
    # 1. 標註藍色區塊的數字 (Contraction)
    labels_blue = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in ax.containers[0]]
    ax.bar_label(
        ax.containers[0],
        labels=labels_blue,
        label_type='center',            # 自動處理垂直置中 # 'center' 會把數字放在藍色柱子的正中央
        rotation=text_rotation,
        fontsize=text_fontsize,
        color="#315a7d",                # 藍底配白字
        #xytext=blue_xytext,
        fontweight='bold',
    )
    
    print(f'[C] labels_blue = {labels_blue}')
    print(f'[C] ax.containers[0] = {ax.containers[0]}')
    
    # 2. 標註紅色區塊的數字 (Expansion)
    labels_red = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in ax.containers[1]]
    ax.bar_label(
        ax.containers[1],
        labels=labels_red,
        label_type='edge',             # 自動處理邊緣貼齊 # 'edge' 會把數字放在紅色柱子的頂端邊緣
        rotation=text_rotation,
        fontsize=text_fontsize,
        color="#46b4af",
        #xy=(3, 5),
        #xytext=red_xytext,
        #textcoords="offset points"
        fontweight='bold',
    )
    
    print(f'[C] labels_red = {labels_red}')
    print(f'[C] ax.containers[1] = {ax.containers[1]}')
    
    
    # 關閉 X 軸的網格線
    ax.grid(False, axis='x')
    # 單獨調整 Y 軸內部網格線（橫線） 的樣式，透明度（alpha）、線條樣式（linestyle） 以及 粗細（linewidth）
    ax.grid(
        True,
        axis='y',            # 只針對 Y 軸內部橫線
        color='gray',        # 線條顏色（預設多為灰色）
        alpha=0.3,           # 1. 透明度：範圍 0.0 ~ 1.0 (0.3 代表很淡、不搶主體視覺)
        linestyle='--',      # 2. 樣式：'--' 為虛線, ':' 為點線, '-' 為實線
        linewidth=0.8        # 3. 粗細：數字越小越細
    )
    
    
    plt.title(
        "Class Distribution and Boundary Harmonization via SAM", pad=15
    )
    plt.xlabel("Category (Class)")
    plt.ylabel("Object Count")
    
    # 將 X 軸標籤的 WASTE_ 彈性替換為 MD_
    ax.set_xticklabels([label.replace("WASTE_", "MD_") for label in count_matrix.index])
    
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Contraction", "Expansion"], loc="upper right")
    
    # 頂部留 15% 的空隙
    totals = count_matrix.sum(axis=1)
    plt.ylim(0, max(totals) * 1.05)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "01_label_distribution_behavior_num_adv.png"), dpi=600
    )
    plt.close()





    # =======================================================================
    # 圖二：BNER 與 TBRR 的雙峰機率密度分佈圖 (KDE of BNER and TBRR)
    # =======================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    df_bner = df[df["p-m_name"] == "BNER"]
    sns.histplot(
        data=df_bner,
        x="p-m_value",
        kde=True,
        color="#4682B4",
        ax=ax1,
        stat="probability",
    )
    ax1.set_title("Background Noise Elimination Ratio (BNER)")
    ax1.set_xlabel("Reduction Percentage (%)")
    ax1.set_ylabel("Probability")

    df_tbrr = df[df["p-m_name"] == "TBRR"]
    sns.histplot(
        data=df_tbrr,
        x="p-m_value",
        kde=True,
        color="#CD5C5C",
        ax=ax2,
        stat="probability",
    )
    ax2.set_title("Target Boundary Recovery Ratio (TBRR)")
    ax2.set_xlabel("Expansion Percentage (%)")
    ax2.set_ylabel("Probability")

    plt.suptitle(
        "Quantitative Evaluation of Annotation Refinement", y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_refinement_kde.png"), dpi=300)
    plt.close()

    # =======================================================================
    # 圖三：BBox 尺寸與長寬比分佈散佈圖 (BBox Scale & Aspect Ratio Diversity)
    # =======================================================================
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="scale",
        y="aspect_ratio",
        hue="label",
        alpha=0.6,
        palette="tab10",
        edgecolor=None,
    )
    plt.axhline(
        1.0, color="gray", linestyle="--", alpha=0.7
    )  # 1.0 代表完美的正方形參考線
    plt.xscale("log")  # 使用對數軸
    plt.title("BBox Structural Diversity: Scale vs. Aspect Ratio", pad=15)
    plt.xlabel("Object Scale (Sqrt of Area in Pixels, Log Scale)")
    plt.ylabel("Aspect Ratio (Width / Height)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Categories")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_scale_aspect_ratio.png"), dpi=300)
    plt.close()

    # =======================================================================
    # 圖四：包含率指標分解小提琴圖 (Decomposed IoU: P_bbox & R_bbox)
    # =======================================================================
    plt.figure(figsize=(8, 6))
    plot_data = pd.melt(
        df,
        id_vars=["label"],
        value_vars=["P_bbox", "R_bbox"],
        var_name="Metric",
        value_name="Ratio",
    )

    sns.violinplot(
        data=plot_data,
        x="Metric",
        y="Ratio",
        hue="Metric",
        split=False,
        inner="quartile",
        palette="Pastel1",
        legend=False,
    )
    plt.title("Decomposed BBox Inclusion Ratios (Geometric Fidelity)", pad=15)
    plt.xlabel("BBox Alignment Metrics")
    plt.ylabel("Inclusion Ratio (0.0 - 1.0)")
    plt.xticks(
        [0, 1],
        [
            r"Precision-like ($P_{bbox}$)" + "\n" + r"[Orig $\cap$ SAM / SAM]",
            r"Recall-like ($R_{bbox}$)" + "\n" r"[Orig $\cap$ SAM / Orig]",
        ],
    )
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_inclusion_ratios.png"), dpi=300)
    plt.close()

    # =======================================================================
    # 圖五：物件中心點歸一化空間分佈熱點圖 (Normalized Spatial Distribution Heatmap)
    # =======================================================================
    plt.figure(figsize=(8, 7))
    # 使用 hexbin 繪製六角格熱點圖
    hb = plt.hexbin(
        df["norm_cx"],
        df["norm_cy"],
        gridsize=25,
        cmap="YlOrRd",
        mincnt=1,
        edgecolors="none",
    )
    cb = plt.colorbar(hb, ax=plt.gca(), label="Object Center Density Count")

    plt.title(
        "Normalized Spatial Distribution of Object Centers", pad=15
    )
    plt.xlabel("Normalized X Coordinate (0.0: Left -> 1.0: Right)")
    plt.ylabel("Normalized Y Coordinate (0.0: Top -> 1.0: Bottom)")
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # 翻轉 Y 軸，使其與影像座標系（左上角為 0,0）一致
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_spatial_heatmap.png"), dpi=300)
    plt.close()

    print(
        f"🎉 所有圖表已成功生成，儲存路徑為: {output_dir}"
    )


if __name__ == "__main__":
    # 使用安全路徑綁定，確保 Linux / Mac 都能執行
    script_dir = Path(__file__).resolve().parent

#    # 預設讀取轉換後的統計 CSV
#    CSV_split = "refinement_10K_20260519-refill2.csv"
    
    # 預設讀取轉換後的統計 CSV
    CSV_split = "refinement_500.csv"
    
    
    print(f'[C] split= {CSV_split}')
    INPUT_CSV = f'{script_dir}/{CSV_split}'
    OUTPUT_IMAGE_FOLDER = script_dir / "eda_report_figures"

    run_eda_pipeline(str(INPUT_CSV), str(OUTPUT_IMAGE_FOLDER))



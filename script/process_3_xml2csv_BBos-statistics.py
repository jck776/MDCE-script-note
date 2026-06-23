#------------------------------------------------------------------------------------------------------------------------#
#
# 2026-05-19
#
# [process]_3_xml2csv_BBos-statistics.py
# 在轉換成CSV格式時，增加計算BBox優化前後差異的計算結果。
# 轉換成CSV格式：
#   欄位：name, hash_id, height, width, label, polygon, SAMBBox, SAM_Area_p-m, p-m_name, p-m_value, P_bbox, R_bbox
#
#   SAM_Area_p-m: 若SAMBBox面積小於原始polygon面積時，設為"-"，若SAMBBox面積大於原始polygon面積時，設為"＋"。
#   p-m_name:若SAMBBox面積小於原始polygon面積時，設為"BNER"，若SAMBBox面積大於原始polygon面積時，設為"TBRR"。
#   p-m_value: 放置SAMBBox於原始polygon的BNER或TBRR計算結果。
#   P_bbox:原始框在 SAM 內的比例 (Precision-like BBox Metric)值。
#   R_bbox:SAM 框在原始框內的比例 (Recall-like BBox Metric)值。
#
#
#   $py \[process\]_3_xml2csv_BBos-statistics.py
#   統計轉換成功！已儲存至：/home/USER/2026_05_19_Refine_BBox_SAM/refinement_10K_20260519-refill2.csv
#
#------------------------------------------------------------------------------------------------------------------------#

import csv
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def calculate_bbox_metrics(poly_points, sam_points):
    """計算 BBox 的幾何指標，包含面積、交集、BNER/TBRR、P_bbox 與 R_bbox"""
    # 解析 原始 polygon 座標 (格式: "x1,y1;x2,y2")
    poly_pts = [list(map(int, pt.split(","))) for pt in poly_points.split(";")]
    p_x1, p_y1 = poly_pts[0][0], poly_pts[0][1]
    p_x2, p_y2 = poly_pts[1][0], poly_pts[1][1]

    # 確保左上角與右下角的順序正確
    p_xmin, p_xmax = min(p_x1, p_x2), max(p_x1, p_x2)
    p_ymin, p_ymax = min(p_y1, p_y2), max(p_y1, p_y2)

    # 解析 優化 SAMBBox 座標
    sam_pts = [list(map(int, pt.split(","))) for pt in sam_points.split(";")]
    s_x1, s_y1 = sam_pts[0][0], sam_pts[0][1]
    s_x2, s_y2 = sam_pts[1][0], sam_pts[1][1]

    s_xmin, s_xmax = min(s_x1, s_x2), max(s_x1, s_x2)
    s_ymin, s_ymax = min(s_y1, s_y2), max(s_y1, s_y2)

    # 1. 計算各自的面積
    area_orig = (p_xmax - p_xmin) * (p_ymax - p_ymin)
    area_sam = (s_xmax - s_xmin) * (s_ymax - s_ymin)

    # 避免除以零的極端狀況
    if area_orig == 0 or area_sam == 0:
        return "-", "ERROR", 0.0, 0.0, 0.0

    # 2. 計算具方向性的面積指標 (SAM_Area_p-m, p-m_name, p-m_value)
    if area_sam < area_orig:
        sam_area_pm = "-"
        pm_name = "BNER"
        # 背景雜訊消除率 (BNER) = ((Area_Orig - Area_SAM) / Area_Orig) * 100%
        pm_value = ((area_orig - area_sam) / area_orig) * 100
    else:
        sam_area_pm = "+"
        pm_name = "TBRR"
        # 目標邊緣補回率 (TBRR) = ((Area_SAM - Area_Orig) / Area_Orig) * 100%
        pm_value = ((area_sam - area_orig) / area_orig) * 100

    # 3. 計算交集區域 (Intersection) 的面積
    inter_xmin = max(p_xmin, s_xmin)
    inter_ymin = max(p_ymin, s_ymin)
    inter_xmax = min(p_xmax, s_xmax)
    inter_ymax = min(p_ymax, s_ymax)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    area_inter = inter_w * inter_h

    # 4. 計算包含率比例 (P_bbox 與 R_bbox)
    p_bbox = area_inter / area_sam
    r_bbox = area_inter / area_orig

    # 將數值四捨五入至小數點後四位，方便閱讀與統計
    return (
        sam_area_pm,
        pm_name,
        round(pm_value, 4),
        round(p_bbox, 4),
        round(r_bbox, 4),
    )


def xml_to_csv_with_stats(xml_path, csv_path):
    # 讀取並解析 XML 檔案
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {xml_path}")
        return
    except ET.ParseError:
        print("錯誤：XML 檔案格式損毀或不正確")
        return

    # 新增學術統計欄位
    headers = [
        "name",
        "hash_id",
        "height",
        "width",
        "label",
        "polygon",
        "SAMBBox",
        "SAM_Area_p-m",
        "p-m_name",
        "p-m_value",
        "P_bbox",
        "R_bbox",
    ]

    rows = []

    # 尋找所有的 <image> 標籤
    for img in root.findall("image"):
        img_info = {
            "name": img.get("name"),
            "hash_id": img.get("hash_id"),
            "height": img.get("height"),
            "width": img.get("width"),
        }

        polygons = img.findall("polygon")
        sambboxes = img.findall("SAMBBox")

        if len(polygons) != len(sambboxes):
            print(
                f"警告：圖片 {img_info['name']} 的 polygon ({len(polygons)}) 與 SAMBBox ({len(sambboxes)}) 數量不匹配！"
            )

        # 逐一將物件組合成 Row，並計算科學指標
        for poly, box in zip(polygons, sambboxes):
            poly_points = poly.get("points")
            sam_points = box.get("points")

            # 呼叫幾何計算核心
            sam_area_pm, pm_name, pm_value, p_bbox, r_bbox = (
                calculate_bbox_metrics(poly_points, sam_points)
            )

            row = {
                "name": img_info["name"],
                "hash_id": img_info["hash_id"],
                "height": img_info["height"],
                "width": img_info["width"],
                "label": poly.get("label"),
                "polygon": poly_points,
                "SAMBBox": sam_points,
                "SAM_Area_p-m": sam_area_pm,
                "p-m_name": pm_name,
                "p-m_value": f"{pm_value}%" if pm_name != "ERROR" else "0.0%",
                "P_bbox": p_bbox,
                "R_bbox": r_bbox,
            }
            rows.append(row)

    # 寫入 CSV 檔案 (使用 utf-8-sig 確保 Windows Excel 打開中文不亂碼)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"統計轉換成功！已儲存至：{csv_path} (共處理 {len(rows)} 個標註物件)"
    )


if __name__ == "__main__":
    # 使用安全路徑綁定方法，避免 Linux 找不到檔案
    script_dir = Path(__file__).resolve().parent

    # 取得參數或使用指定的預設檔名
    xml_name = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else "refinement_10K_20260519-refill2.xml"
    )
    csv_name = xml_name.replace(".xml", ".csv")

    xml_absolute_path = script_dir / xml_name
    csv_absolute_path = script_dir / csv_name

    xml_to_csv_with_stats(xml_absolute_path, csv_absolute_path)

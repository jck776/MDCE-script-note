import os
import sys
from pathlib import Path
import cv2
import pandas as pd


def draw_dotted_rect(img, pt1, pt2, color, thickness=1, gap=4):
    """因為 OpenCV 沒有內建虛線矩形，這裡用簡單的線段模擬虛線框，增加原始框的辨識度"""
    x1, y1 = pt1
    x2, y2 = pt2

    # 橫線
    for x in range(x1, x2, gap * 2):
        cv2.line(
            img,
            (x, y1),
            (min(x + gap, x2), y1),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            img,
            (x, y2),
            (min(x + gap, x2), y2),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    # 直線
    for y in range(y1, y2, gap * 2):
        cv2.line(
            img,
            (x1, y),
            (x1, min(y + gap, y2)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            img,
            (x2, y),
            (x2, min(y + gap, y2)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def visualize_annotations(csv_path, image_dir, output_dir):
    # 讀取 CSV 檔案
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到 CSV 檔案 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # WAST_1 ~WAST_20 替換成 MD_1 ~ MD_20 #
    # 在你跑迴圈或畫圖「之前」，先執行這一行：
    df["label"] = df["label"].astype(str).str.replace("WASTE_", "MD_", regex=False)


    os.makedirs(output_dir, exist_ok=True)

    # 定義色彩與線條細節 (OpenCV 使用 BGR 格式)
    COLOR_POLY = (0, 0, 255)  # 原始框：深紅色
    COLOR_SAM = (0, 255, 0)  # 優化框：螢光綠 (主體)
    COLOR_TEXT = (255, 255, 255)  # 文字：純白

    THICKNESS_POLY = 1  # 原始框較細
    THICKNESS_SAM = 3  # 優化框加粗，強調主體

    # 按照片名稱分組處理，避免重複讀取同一張圖
    grouped = df.groupby("name")

    for img_name, group in grouped:
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"警告：找不到影像檔案 {img_path}，跳過此張。")
            continue

        # 讀取影像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：無法讀取影像 {img_name}")
            continue

        # 繪製該張圖片的所有標註
        for _, row in group.iterrows():
            label = str(row["label"])

            # 解析 原始 polygon 座標 (格式: "x1,y1;x2,y2")
            poly_pts = [
                list(map(int, pt.split(","))) for pt in row["polygon"].split(";")
            ]
            poly_p1, poly_p2 = tuple(poly_pts[0]), tuple(poly_pts[1])

            # 解析 優化 SAMBBox 座標 (格式: "x1,y1;x2,y2")
            sam_pts = [
                list(map(int, pt.split(","))) for pt in row["SAMBBox"].split(";")
            ]
            sam_p1, sam_p2 = tuple(sam_pts[0]), tuple(sam_pts[1])

            # 1. 繪製原始框 (使用細虛線，降低視覺干擾)
            draw_dotted_rect(
                img, poly_p1, poly_p2, COLOR_POLY, thickness=THICKNESS_POLY
            )

            # 2. 繪製優化後的 SAMBBox (實線加粗，高對比)
            cv2.rectangle(
                img, sam_p1, sam_p2, COLOR_SAM, thickness=THICKNESS_SAM
            )

            # 3. 繪製帶有填充底色的 Label (僅在原始框左上角繪製一次)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2

            # 計算文字寬高，用來製作完美的背景遮罩矩形
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )

            # 計算底色矩形的位置 (置於原始框左上角上方，若出界則往下壓)
            txt_bg_p1 = (poly_p1[0], max(0, poly_p1[1] - text_h - 6))
            txt_bg_p2 = (poly_p1[0] + text_w + 6, max(text_h + 6, poly_p1[1]))

            # 畫出填滿的藍/紅色背景 (與原始框同色，建立視覺聯結)
            cv2.rectangle(img, txt_bg_p1, txt_bg_p2, COLOR_POLY, cv2.FILLED)

            # 寫上白字標籤
            text_position = (txt_bg_p1[0] + 3, txt_bg_p2[1] - 3)
            cv2.putText(
                img,
                label,
                text_position,
                font,
                font_scale,
                COLOR_TEXT,
                text_thickness,
                lineType=cv2.LINE_AA,
            )

        # 儲存繪製完成的圖片
        output_path = os.path.join(output_dir, f"vis_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"已生成視覺化結果：{output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    # 請根據您的環境修改以下路徑：
    CSV_FILE = script_dir / "refinement_10K_20260519-refill2.csv"  # 您的 CSV 檔案路徑
    IMAGE_DIR = script_dir / "10K/"  # 存放原始 .jpg 圖片的資料夾
    OUTPUT_DIR = script_dir / "BBox_plot/"  # 輸出結果資料夾

#    # 請根據您的環境修改以下路徑：
#    CSV_FILE = script_dir / "refinement_500.csv"  # 您的 CSV 檔案路徑
#    IMAGE_DIR = script_dir / "500/"  # 存放原始 .jpg 圖片的資料夾
#    OUTPUT_DIR = script_dir / "BBox_plot_500/"  # 輸出結果資料夾
    
    
    # 執行視覺化
    visualize_annotations(str(CSV_FILE), str(IMAGE_DIR), str(OUTPUT_DIR))

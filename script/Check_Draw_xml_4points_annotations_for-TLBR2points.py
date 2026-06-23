"""
[MDCE_Script] 2026-06-23

Check BBox of single image from original pre_10K.xml.  

▶ python3 Check_Draw_xml_4points_annotations_for-TLBR2points.py --xml pre_10K.xml --name 20221104_新北市_北海岸淺水灣-090.jpg
"""


import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import re
import argparse



def draw_xml_annotations_TRBL_box(xml_file, image_dir, target_image_name):
    """
    讀取 XML，根據檔名搜尋標註並繪製顯示
    """
    # 1. 檢查圖片檔案是否存在
    img_path = os.path.join(image_dir, target_image_name)
    if not os.path.exists(img_path):
        print(f"[-] 錯誤：在資料夾 '{image_dir}' 中找不到圖片 '{target_image_name}'")
        return

    # 2. 載入圖片 (支援中文路徑)
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[-] 錯誤：無法讀取圖片 '{target_image_name}'")
        return

    # 3. 解析 XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 尋找名稱符合的 <image> 標籤
        target_node = None
        for img_node in root.findall('image'):
            if img_node.get('name') == target_image_name:
                target_node = img_node
                break
        
        if target_node is None:
            print(f"[-] 警告：在 XML 中找不到檔名為 '{target_image_name}' 的標註資料")
            return

        # 4. 遍歷該圖片下的所有標註
        polygons = target_node.findall('polygon')
        print(f"[+] 找到 {len(polygons)} 個標註項目於 '{target_image_name}'")

        for i, poly in enumerate(polygons):
            label = poly.get('label', 'Unknown')
            points_str = poly.get('points')
            
            if not points_str:
                continue

            # 解析座標字串
            raw_coords = re.split(',|;', points_str)
            coords = [int(float(c)) for c in raw_coords if c.strip()]
            
            # 轉換為 OpenCV 格式 (N, 1, 2)
            pts = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
            print(f"[check] pts : {pts}")
            print(f"[check] pts : {pts}, shape: {pts.shape}")
            print(f"[check] 左上 TL : {pts[0][0]}")
            print(f"[check] 右下 BR : {pts[1][0]}")
            
            # 繪製邊框 (鮮綠色)
            #cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.rectangle(image, pts[0][0], pts[1][0], (0, 255, 0), 4)
            
            # 在框上方標註 Label
            # 取第一個點作為文字位置
            text_pos = (pts[0][0][0], pts[0][0][1] + 30)
            cv2.putText(image, f"#{i} {label}", text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 4, cv2.LINE_AA)

        # 5. 顯示結果 (自動縮放以適應螢幕)
        h, w = image.shape[:2]
        max_height = 900 # 設定顯示的最大高度
        if h > max_height:
            scale = max_height / h
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        cv2.imshow(f"Preview: {target_image_name}", image)
        print("[*] 預覽視窗已開啟，按任意鍵關閉...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ET.ParseError:
        print(f"[-] 錯誤：XML 檔案 '{xml_file}' 格式損壞或解析失敗")
    except Exception as e:
        print(f"[-] 發生非預期錯誤: {e}")
        
        
def draw_xml_annotations_polylines(xml_file, image_dir, target_image_name):
    """
    讀取 XML，根據檔名搜尋標註並繪製顯示
    """
    # 1. 檢查圖片檔案是否存在
    img_path = os.path.join(image_dir, target_image_name)
    if not os.path.exists(img_path):
        print(f"[-] 錯誤：在資料夾 '{image_dir}' 中找不到圖片 '{target_image_name}'")
        return

    # 2. 載入圖片 (支援中文路徑)
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[-] 錯誤：無法讀取圖片 '{target_image_name}'")
        return

    # 3. 解析 XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 尋找名稱符合的 <image> 標籤
        target_node = None
        for img_node in root.findall('image'):
            if img_node.get('name') == target_image_name:
                target_node = img_node
                break
        
        if target_node is None:
            print(f"[-] 警告：在 XML 中找不到檔名為 '{target_image_name}' 的標註資料")
            return

        # 4. 遍歷該圖片下的所有標註
        polygons = target_node.findall('polygon')
        print(f"[+] 找到 {len(polygons)} 個標註項目於 '{target_image_name}'")

        for i, poly in enumerate(polygons):
            label = poly.get('label', 'Unknown')
            points_str = poly.get('points')
            
            if not points_str:
                continue

            # 解析座標字串
            raw_coords = re.split(',|;', points_str)
            coords = [int(float(c)) for c in raw_coords if c.strip()]
            
            # 轉換為 OpenCV 格式 (N, 1, 2)
            pts = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
            print(f"[check] pts : {pts}, shape: {pts.shape}")
            print(f"[check] 左上 TL : {pts[0][0]}")
            print(f"[check] 右下 BR : {pts[3][0]}")
            
            # 繪製邊框 (鮮綠色)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # 在框上方標註 Label
            # 取第一個點作為文字位置
            text_pos = (pts[0][0][0], pts[0][0][1] - 10)
            cv2.putText(image, f"#{i} {label}", text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # 5. 顯示結果 (自動縮放以適應螢幕)
        h, w = image.shape[:2]
        max_height = 900 # 設定顯示的最大高度
        if h > max_height:
            scale = max_height / h
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        cv2.imshow(f"Preview: {target_image_name}", image)
        print("[*] 預覽視窗已開啟，按任意鍵關閉...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ET.ParseError:
        print(f"[-] 錯誤：XML 檔案 '{xml_file}' 格式損壞或解析失敗")
    except Exception as e:
        print(f"[-] 發生非預期錯誤: {e}")

if __name__ == "__main__":
    # 設定 CLI 參數
    parser = argparse.ArgumentParser(description="MDImageNet 標註檢查工具 (BBox Checker)")
    
    # 必填參數
    parser.add_argument("--xml", required=True, help="標註 XML 檔案路徑 (例如: annotations_v2.xml)")
    parser.add_argument("--name", required=True, help="要檢查的影像檔名 (例如: 20220909_新北市-金山-F3846.jpg)")
    
    # 選填參數
    parser.add_argument("--dir", default="10K/", help="影像檔案所在的資料夾路徑 (預設為當前目錄 '.')")

    args = parser.parse_args()

    # 執行繪製功能
    
    # Poly lines   #
    draw_xml_annotations_polylines(args.xml, args.dir, args.name)
    
    # Bounding box #
    draw_xml_annotations_TRBL_box(args.xml, args.dir, args.name)

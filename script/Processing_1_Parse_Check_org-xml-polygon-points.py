#---------------------------------------------------------------------------------------------------------------#
#
#
#   Processing_1_Parse_Check_org-xml-polygon-points.py
#       (1) 用於檢查原始標注檔[dataset_10000_update_0912.xml]，將小部分非矩形、超過4點的點位修正後，另存為新的pre_10K.xml。
#       (2) 原檔是以polygon四點儲存，預處理過程中會將四點但非標準box與超過四點者，繪製顯示並列印出修正後的左上右下座標，兩點重複一遍來標示。
#       (3) 增加hash_id到pre_10K.xml。
#       (4) SAM refine the bounding box. { 由 [Processing]_2_SAM-Refine-Box.py 接續 產出10K.xml -> VOC }
#
#   ▶ python3 Processing_1_Parse_Check_org-xml-polygon-points.py dataset_10000_update_0912.xml
#   成功！優化後的標註已儲存至: pre_10K.xml
#
#
#---------------------------------------------------------------------------------------------------------------#

import xml.etree.ElementTree as ET
import sys
import re
import cv2
import numpy as np
import hashlib


def get_12char_hash_blake2(filename, salt="md#2023@CE#2026"):
    # 將檔名與鹽值組合
    data = (filename + salt).encode('utf-8')
    # digest_size=6 代表輸出 6 bytes = 12 個十六進位字元
    h = hashlib.blake2b(data, digest_size=6)
    return h.hexdigest()

def draw_annotation(image, points_str, color=(0, 255, 0), thickness=2):
    """
    在影像上繪製任意點數的多邊形或邊界框
    
    :param image: 原始影像 (numpy array)
    :param points_str: 座標字串 (例如 "x1,y1;x2,y2..." 或 "x1,y1,x2,y2...")
    :param color: 線條顏色 (BGR 格式，預設為綠色)
    :param thickness: 線條粗細
    :return: 繪製完成後的影像
    """
    if not points_str or not isinstance(points_str, str):
        return image

    # 1. 解析字串並轉換為數值列表
    try:
        # 使用正規表達式拆分逗號或分號
        raw_coords = re.split(',|;', points_str)
        # 過濾空字串並轉為整數
        coords = [int(float(c)) for c in raw_coords if c.strip()]
        
        if len(coords) < 4 or len(coords) % 2 != 0:
            print(f"Warning: 無效的座標數量 {len(coords)}")
            return image

        # 2. 轉換為 OpenCV 要求的格式: (點數, 1, 2) 的 int32 numpy array
        points = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)

        # 3. 繪製多邊形
        # isClosed=True 代表會自動將最後一點連回第一點
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        
        # 選配：繪製點的序號 (Debug 用，若不需要可註解掉)
        # for i, pt in enumerate(points):
        #    cv2.putText(image, str(i), tuple(pt[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    except Exception as e:
        print(f"繪製標註時發生錯誤: {e}")

    return image

def is_perfect_bbox(points_list):
    """
    檢查 4 個點是否為完美的軸對齊邊界框 (Axis-Aligned Bounding Box)
    邏輯：完美的矩形在所有頂點中，只會存在 2 個唯一的 X 值與 2 個唯一的 Y 值。
    """
    if len(points_list) != 4:
        return False
    
    # 提取所有點的 x 和 y 座標
    # points_list 格式範例: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    xs = [pt[0] for pt in points_list]
    ys = [pt[1] for pt in points_list]
    
    # 使用 set 取得不重複的座標數量
    unique_xs = set(xs)
    unique_ys = set(ys)
    
    # 如果剛好都是 2 個，則為完美的 BBox
    return len(unique_xs) == 2 and len(unique_ys) == 2


def bbox_from_polygon(ply):
    """
    從任意點數的多邊形座標字串中提取最小外接矩形 (Bounding Box)
    輸入格式範例: "x1,y1;x2,y2;x3,y3..." 或 "x1,y1,x2,y2..."
    回傳: xmin, ymin, xmax, ymax
    """
    # 1. 使用正規表達式拆分所有數字 (支援逗號或分號分隔)
    coords = re.split(',|;', ply)
    
    # 2. 轉換為整數陣列
    # 過濾掉可能因為末尾分隔符產生的空字串
    coords = np.array([int(c) for c in coords if c.strip()], dtype="int")
    
    # 3. 將一維陣列重塑為 (N, 2)，其中 N 是點的數量
    # 每一橫列代表一個點 [x, y]
    points = coords.reshape(-1, 2)
    
    # 4. 根據公式取極值
    # points[:, 0] 是所有的 x 座標, points[:, 1] 是所有的 y 座標
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    
    # [CK] Debug 資訊，方便確認轉換結果
    # print(f"[CK] Original points count: {len(points)}")
    # print(f"[CK] BBox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    
    return xmin, ymin, xmax, ymax

def parse_polygon_points(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # temp hash table
        all_hashes = set()
        
        for image in root.findall('image'):
            image_name = image.get('name')
            
            # Hash new hex filename
            if image_name:
                # 根據檔名產生 12 字元 Hash
                hid = get_12char_hash_blake2(image_name)

                # hash collision check
                if hid in all_hashes:
                    # 發生碰撞！必須加上序號或額外處理
                    raise ValueError(f"CRITICAL ERROR: Hash collision detected for {file}!")
                all_hashes.add(hid)
                
                # 使用 .set() 新增 hash_id 屬性
                image.set('hash_id', hid)
                             
                # 印出結果確認
                print(f"Added hash_id='{hid}' for image '{image_name}'")
                
            
            for polygon in image.findall('polygon'):
                points_str = polygon.get('points')
                label_str  = polygon.get('label')
                if not points_str:
                    continue
                
                # 解析座標字串為數值列表 [[x, y], [x, y], ...]
                # 支援分號與逗號分隔
                raw_coords = re.split(',|;', points_str)
                points = []
                for i in range(0, len(raw_coords), 2):
                    if raw_coords[i].strip():
                        points.append([float(raw_coords[i]), float(raw_coords[i+1])])
                
                point_count = len(points)
                
                # 1. 檢查數量異常的情況 ploygon 低於4點或多過4點時
                if point_count > 4 or point_count < 4:
                    print(f"[Count Warning] Image: {image_name} | Points: {point_count}")
                    
                    
                    # 計算polygon最大水平垂直的矩形並印出
                    xmin, ymin, xmax, ymax = bbox_from_polygon(points_str)
                    print(f"{label_str} [xmin, ymin, xmax, ymax]= {xmin},{ymin};{xmax},{ymax}") #;{xmin},{ymin};{xmax},{ymax}")
                    
                    new_points_str = f"{xmin},{ymin};{xmax},{ymax}" # 改成僅存左上右下 ;{xmin},{ymin};{xmax},{ymax}"
                    # 寫回 Element 物件中
                    polygon.set('points', new_points_str)
                    
                    # 繪製檢查
                    img = cv2.imread('10K/' + image_name)
                    test_img = draw_annotation(img, points_str, color=(0, 255, 255), thickness=2) # 黃色
                    # 顯示結果
                    cv2.imshow("Annotation Test", test_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    
                # 2. 當數量等於 4 時，檢查是否為完美的 BBox
                elif point_count == 4:
                    if is_perfect_bbox(points):
                        # print(f"[Perfect BBox] Image: {image_name}") # 選配：印出正確的
                        #pass
                        
                        # 雖然是正確box型態，但是以4點紀錄，會有8個值。左上右下需要取頭尾。先行過濾為佳。
                        print(f"[Perfect BBox 4 points] Image: {image_name} | 4 points standard BBox")
                         # 計算polygon最大水平垂直的矩形並印出作為修改成10K.xml依據
                        xmin, ymin, xmax, ymax = bbox_from_polygon(points_str)
                        print(f"{label_str} [xmin, ymin, xmax, ymax]= {xmin},{ymin};{xmax},{ymax}") #;{xmin},{ymin};{xmax},{ymax}")
                        
                        new_points_str = f"{xmin},{ymin};{xmax},{ymax}" # 改成僅存左上右下 ;{xmin},{ymin};{xmax},{ymax}"
                        # 寫回 Element 物件中
                        polygon.set('points', new_points_str)
                        
                        
#                        # 繪製檢查 (因為是正常box之後可以關閉顯示)
#                        img = cv2.imread('10K/' + image_name)
#                        #test_img = draw_annotation(img, points_str, color=(0, 255, 255), thickness=2) # 黃色
#                        test_img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#                        # 顯示結果
#                        cv2.imshow("Annotation Test", test_img)
#                        cv2.waitKey(0)
#                        cv2.destroyAllWindows()
                        
                    else:
                        print(f"[Format Warning] Image: {image_name} | 4 points but NOT a standard BBox (possibly tilted or quadrilateral)")

                        # 計算polygon最大水平垂直的矩形並印出作為修改成10K.xml依據
                        xmin, ymin, xmax, ymax = bbox_from_polygon(points_str)
                        print(f"{label_str} [xmin, ymin, xmax, ymax]= {xmin},{ymin};{xmax},{ymax}") #;{xmin},{ymin};{xmax},{ymax}")
                        
                        new_points_str = f"{xmin},{ymin};{xmax},{ymax}" # 改成僅存左上右下 ;{xmin},{ymin};{xmax},{ymax}"
                        # 寫回 Element 物件中
                        polygon.set('points', new_points_str)
                        
                        
                        # 繪製檢查
                        img = cv2.imread('10K/' + image_name)
                        test_img = draw_annotation(img, points_str, color=(0, 255, 255), thickness=2) # 黃色
                        # 顯示結果
                        cv2.imshow("Annotation Test", test_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    
    
        #  另存新檔 (設定 encoding 確保中文檔名不會亂碼)
        output_xml="pre_500.xml" #"pre_10K.xml"
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        print(f"成功！優化後的標註已儲存至: {output_xml}")
    
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {xml_file}")
    except ET.ParseError:
        print(f"錯誤：XML解析失敗")

if __name__ == "__main__":
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "500.xml" #"dataset_10000_update_0912.xml" 
    parse_polygon_points(xml_file)

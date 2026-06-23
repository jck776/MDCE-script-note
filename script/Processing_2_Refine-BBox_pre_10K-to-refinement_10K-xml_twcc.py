#---------------------------------------------------------------------------------------------------------------#
#
#
#   [Processing]_2_Refine-BBox_pre_10K-to10K-xml.py
#       (1) 來源 [pre_10K.xml] 已經過修正，均為正確的[左上,右下, 左上,右下] 重複一次格式是配合步驟 [Processing]_3 的處理格式。
#       (1) SAM refine the bounding box. { 由 [Processing]_2_Refine-BBox_pre_10K-to10K-xml.py 接續 產出10K.xml -> VOC }
#       (2)
#
#
#
#   ▶ python3 "[Processing]_2_Refine-BBox_pre_10K-to10K-xml.py" pre_10K.xml
#   成功！優化後的標註已儲存至: 10K.xml
#
#
#---------------------------------------------------------------------------------------------------------------#


import xml.etree.ElementTree as ET
import sys
import re
import cv2
import numpy as np
from ultralytics import SAM
from tqdm import tqdm


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

def BBox_Refinement(img_dir, xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # SAM model #
        model = SAM("sam2.1_l.pt") # t,s,b,l
        # Display model information (optional)
        model.info()
        
        for image in tqdm(root.findall('image')):
            image_name = image.get('name')
            
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
                
                xmin, ymin, xmax, ymax = bbox_from_polygon(points_str)
                #print(f"[Org] {label_str} [xmin, ymin, xmax, ymax]= {xmin},{ymin};{xmax},{ymax}")
                
                #                                         #
                # pre_10K.xml 已經過[處理1階]，不需要再重複檢查。 #
                #                                         #
                
                if point_count == 2: #儲存都改成左上下，不補滿為原始數量了。
                    # Refine BBox by SAM2.1 #
                    # Run inference with bboxes prompt
                    results = model(img_dir + "/" + image_name, bboxes=[xmin, ymin, xmax, ymax])
                    
                    # 安全防護：檢查 SAM 是否有成功輸出分割結果
                    if not results or len(results[0].boxes) == 0:
                        print(f"[-] 警告: SAM 無法在 {image_name} 中優化該目標，跳過此物件。")
                        continue
                    
                    # 需要Qt, xcb來繪圖
                    #sam_plot = results[0].plot()
                    
                    
                    # 原BBox
                    print(f"[Org] {label_str} [xmin, ymin, xmax, ymax]=     {xmin},{ymin};{xmax},{ymax}")
                    
                    # SAM BBOX
                    for r in results:
                        #print(r.boxes.xyxy.numpy())  # print the Boxes object containing the detection bounding boxes
                        sam_box = r.boxes.xyxy.cpu().numpy()
                        #print(sam_box, sam_box[0][0])
                        Sxmin, Symin, Sxmax, Symax = int(sam_box[0][0]), int(sam_box[0][1]), int(sam_box[0][2]), int(sam_box[0][3])
                        print(f"[SAM] {label_str} [Sxmin, Symin, Sxmax, Symax]= {Sxmin},{Symin};{Sxmax},{Symax}")
                        
                        # 使用 ET.SubElement 在當前 image 節點下建立新標籤 <SAMBBox>
                        sambbox = ET.SubElement(image, 'SAMBBox')
                        # 設定新標籤的屬性值
                        sambbox.set('label', label_str)
                        sambbox.set('points', f"{Sxmin},{Symin};{Sxmax},{Symax}")
                        
                else:
                    print(f"[Format Warning] Image: {image_name} | NOT 2 points standard BBox!")
    
        #  另存新檔 (設定 encoding 確保中文檔名不會亂碼)
        # 檢查檔名中是否有底線 "_"
        if "_" not in xml_file:
            print(
            f"錯誤：輸入的檔名 '{input_filename}' 不包含底線 '_'，無法進行截斷。 pre_xxx.xml"
            )
        # 使用底線進行切割
        # stem_name.split('_') 會把 "pre_500" 切成 ["pre", "500"]
        parts = xml_file.split("_")

        # 取出底線後面的部分（即數字 "500"）
        number_part = parts[-1]

        # 組合新的檔名：固定前綴 "refinement_" + 數字 + 原本的副檔名 ".xml"
        output_filename = f"refinement_{number_part}"

        output_xml = output_filename #"refinement_500.xml" #"refinement_10K.xml"
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        print(f"成功！緊縮優化後的標註已儲存至: {output_xml}")
    
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {xml_file}")
    except ET.ParseError:
        print(f"錯誤：XML解析失敗")




if __name__ == "__main__":
#    xml_file = sys.argv[1] if len(sys.argv) > 1 else "pre_10K.xml"

    img_dir = "500" # "10K"
    xml_file = f"pre_{img_dir}.xml"
    BBox_Refinement(img_dir, xml_file)


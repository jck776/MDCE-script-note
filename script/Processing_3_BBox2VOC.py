import os
import re
import cv2
import xmltodict
from PIL import Image
from tqdm import tqdm

# ==================== 設定區 ====================
CONF_SET = "val"  # 可切換 "train" 或 "val"
SHOW_IMAGE = True    # 是否跳出 OpenCV 視窗供人類檢查 (True=顯示, False=不顯示)
JPEG_QUALITY = 80    # 複製後新照片的影像品質設定 (0-100) 同時去除EXIF資訊與使用80%品質壓縮

if CONF_SET == "val":
    PATH_TO_XML = 'refinement_500.xml'
    PATH_DIR = 'MD2023_CE/MDCE_VOC/evl/Annotations_500_xml'
    ImageSets = "MD2023_CE/MDCE_VOC/evl/ImageSets/val.txt"
    JPG_DIR = "500"
    JPEG_IMAGES_DIR = "MD2023_CE/MDCE_VOC/evl/JPEGImages_500"
elif CONF_SET == "train":
    PATH_TO_XML = 'refinement_10K_20260519-refill2.xml'
    PATH_DIR = 'MD2023_CE/MDCE_VOC/train/Annotations_10K_xml'
    ImageSets = "MD2023_CE/MDCE_VOC/train/ImageSets/train.txt"
    JPG_DIR = "10K"
    JPEG_IMAGES_DIR = "MD2023_CE/MDCE_VOC/train/JPEGImages_10K"

# 自動建立所有相關的深度目錄 (含 ImageSets 的父目錄)
os.makedirs(PATH_DIR, exist_ok=True)
os.makedirs(JPEG_IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ImageSets), exist_ok=True)
# ================================================

def ensure_list(obj):
    """確保讀取出來的標籤永遠是 list 格式"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]

def parse_sambbox_points(points_str):
    """解析 SAMBBox 的 points 格式 ("xmin,ymin;xmax,ymax")"""
    coords = [int(x) for x in re.split(r',|;', points_str)]
    if len(coords) == 4:
        return coords[0], coords[1], coords[2], coords[3]
    else:
        raise ValueError(f"無法解析的 SAMBBox 座標格式: {points_str}")

def draw_bounding_box(image, box, label):
    """在影像上繪製 Bounding Box 與標籤文字，方便人類檢查"""
    xmin, ymin, xmax, ymax = box
    color = (0, 0, 255)  # 紅色
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    text_ymin = max(ymin, text_size[1] + 10)
    cv2.rectangle(image, (xmin, text_ymin - text_size[1] - 4), (xmin + text_size[0] + 6, text_ymin + 4), color, -1)
    cv2.putText(image, label, (xmin + 3, text_ymin), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image

def write_pascal_voc_xml(output_dir, folder_name, hash_id, width, height, bbox_list):
    """建立 PASCAL VOC XML 檔案"""
    xml_path = os.path.join(output_dir, f"{hash_id}.xml")
    
    xml_content = f"""<annotation>
    <folder>{os.path.basename(folder_name)}</folder>
    <filename>{hash_id}.jpg</filename>
    <path>{hash_id}.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>"""
    
    for box in bbox_list:
        label, xmin, ymin, xmax, ymax = box
        xml_content += f"""
    <object>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>"""
        
    xml_content += "\n</annotation>"
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)


def dump_id_name_to_JPEGImages(src_jpg_path, dst_jpg_path, quality=95):
    """讀取原始影像，並以指定的 JPEG 品質存入新路徑與新檔名"""
    # 使用 Pillow 打開影像
    with Image.open(src_jpg_path) as img:
        # 建立一個只包含純像素的新影像物件
        data = list(img.getdata())
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(data)
        
        # 取得副檔名格式，並統一轉成大寫（例如: 'JPEG', 'PNG'）
        img_format = img.format if img.format else "JPEG"
        
        if clean_img is not None:
            # 這裡設定壓縮品質與啟用優化
            clean_img.save(
                dst_jpg_path,
                format=img_format,
                quality=quality,  # 設定品質 (1-95，建議 75-85 之間平衡最好)
                optimize=True,  # 關鍵：開啟哈夫曼樹優化，大幅縮減檔案體積
            )
        else:
            # 如果是 PNG 等不支援 quality 的格式，直接正常儲存
            clean_img.save(output_path, format=img_format)



# ==================== 主程式執行 ====================
print(f"正在讀取並解析主 XML 檔案: {PATH_TO_XML}...")
with open(PATH_TO_XML, 'r', encoding='utf-8') as file:
    file_data = file.read()
    dict_data = xmltodict.parse(file_data)

images_data = ensure_list(dict_data["annotations"]["image"])
print(f"總共偵測到 {len(images_data)} 張影像。開始進行轉換與檢查...")

# 用來儲存所有要寫入 ImageSets txt 的行資料
imagesets_lines = []

for img_node in tqdm(images_data):
    img_name = img_node["@name"]
    hash_id = img_node["@hash_id"]
    width = img_node["@width"]
    height = img_node["@height"]
    
    src_jpg_path = os.path.join(JPG_DIR, img_name)
    dst_jpg_path = os.path.join(JPEG_IMAGES_DIR, f"{hash_id}.jpg")
    
    # 紀錄至 txt 對應清單
    imagesets_lines.append(f"{hash_id}.jpg {hash_id}.xml")
    
    # 獲取所有的 SAMBBox 物件
    sambbox_nodes = ensure_list(img_node.get('SAMBBox', []))
    
    bbox_list = []
    check_image = cv2.imread(src_jpg_path) if SHOW_IMAGE else None
    
    for box_node in sambbox_nodes:
        label = box_node['@label']
        points_str = box_node['@points']
        
        xmin, ymin, xmax, ymax = parse_sambbox_points(points_str)
        bbox_list.append((label, xmin, ymin, xmax, ymax))
        
        if check_image is not None:
            check_image = draw_bounding_box(check_image, (xmin, ymin, xmax, ymax), label)
            
    # 1. 寫入新版 PASCAL VOC XML
    write_pascal_voc_xml(PATH_DIR, PATH_DIR, hash_id, width, height, bbox_list)
    
    # 2. 複製並優化影像品質
    dump_id_name_to_JPEGImages(src_jpg_path, dst_jpg_path, quality=JPEG_QUALITY)
    
    # 3. 人類檢查視覺化展示
    if SHOW_IMAGE and check_image is not None:
        display_img = cv2.resize(check_image, (800, int(800 * int(height) / int(width))))
        cv2.imshow("Human Verification (SAMBBox)", display_img)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            print("\n使用者已手動關閉人類檢查視窗，後續將自動處理完畢。")
            SHOW_IMAGE = False
            cv2.destroyAllWindows()

if SHOW_IMAGE:
    cv2.destroyAllWindows()

# 4. 寫入 ImageSets 列表文字檔
print(f"正在寫入 ImageSets 檔案: {ImageSets}...")
with open(ImageSets, 'w', encoding='utf-8') as txt_file:
    txt_file.write("\n".join(imagesets_lines) + "\n")

print("\n🎉 所有轉換工作皆已完成！")

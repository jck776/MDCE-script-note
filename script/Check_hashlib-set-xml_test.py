import xml.etree.ElementTree as ET
import hashlib

def get_12char_hash_blake2(filename, salt="md#2023@CE#2026"):
    """
    計算檔名的 12 字元 Hash (blake2b 方法)
    """
    data = (filename + salt).encode('utf-8')
    h = hashlib.blake2b(data, digest_size=6)
    return h.hexdigest()

def add_hash_id_to_xml(input_xml, output_xml):
    try:
        # 1. 解析 XML
        tree = ET.parse(input_xml)
        root = tree.getroot()
        
        # 2. 遍歷所有的 <image> 標籤
        for image in root.findall('image'):
            image_name = image.get('name')
            
            if image_name:
                # 3. 根據檔名產生 12 字元 Hash
                hid = get_12char_hash_blake2(image_name)
                
                # 4. 使用 .set() 新增 hash_id 屬性
                image.set('hash_id', hid)
                
                # [Debug] 印出結果確認
                print(f"Added hash_id='{hid}' for image '{image_name}'")

        # 5. 儲存修改後的 XML
        # 使用 xml_declaration=True 確保檔案開頭有 <?xml...?>
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        print(f"\n[成功] 新的 XML 已儲存至: {output_xml}")

    except Exception as e:
        print(f"[錯誤] 處理 XML 時發生問題: {e}")

# --- 執行 ---
if __name__ == "__main__":
    # 輸入與輸出檔名
    input_file = "annotations.xml"
    output_file = "annotations_with_hash.xml"
    
    add_hash_id_to_xml(input_file, output_file)

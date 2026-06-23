import os
import xml.etree.ElementTree as ET
from pathlib import Path

def replace_waste_to_md(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: 資料夾 {root_dir} 不存在！")
        return

    xml_files = list(root_path.glob("**/*.xml"))
    print(f"🎯 找到 {len(xml_files)} 個 XML 檔案，開始處理...")

    modify_count = 0
    file_count = 0

    for xml_file in xml_files:
        try:
            # 解析 XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            is_file_changed = False

            # 尋找所有 object底下的 name 標籤
            for obj in root.findall('object'):
                name_node = obj.find('name')
                if name_node is not None and name_node.text:
                    # 如果包含 WASTE_，則進行替換
                    if "WASTE_" in name_node.text:
                        old_name = name_node.text
                        new_name = old_name.replace("WASTE_", "MD_")
                        name_node.text = new_name
                        is_file_changed = True
                        modify_count += 1

            # 如果該檔案有被修改，才寫回檔案（保留原本的 UTF-8 編碼）
            if is_file_changed:
                tree.write(xml_file, encoding='utf-8', xml_declaration=False)
                file_count += 1

        except ET.ParseError:
            print(f"⚠️ 警告: {xml_file} 解析失敗，跳過該檔案。")
        except Exception as e:
            print(f"❌ 處理 {xml_file} 時發生錯誤: {e}")

    print(f"\n==========================================")
    print(f"✅ 處理完成！")
    print(f"📁 總共修改了 {file_count} 個 XML 檔案")
    print(f"🏷️ 總共替換了 {modify_count} 個標籤名稱")
    print(f"==========================================")

if __name__ == "__main__":
    # 指定你的資料集根目錄名稱
    target_directory = "MDCE_VOC" 
    replace_waste_to_md(target_directory)

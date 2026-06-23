"""
[MDCE_Script] 2026-06-23

Check image's EXIF and remove it.  

▶ python3 Check_image_exif_dir.py ./DIR/


#========================================
#📊 掃描報告完成
#========================================
#總計掃描影像檔案數: 10000
#內藏中繼資料的檔案數: 1580
#========================================

#========================================
#📊 掃描報告完成
#========================================
#總計掃描影像檔案數: 500
#內藏中繼資料的檔案數: 43
#========================================


"""

import argparse
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

# 定義支援的影像副檔名
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def scan_folder_metadata(folder_path, recursive=False):
    target_dir = Path(folder_path)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"❌ 錯誤：指定的路徑 '{folder_path}' 不是一個有效的資料夾。")
        return

    print(f"========================================")
    print(f"🔍 開始掃描資料集目錄: {target_dir.resolve()}")
    print(f"🔄 掃描模式: {'包含子資料夾 (遞迴)' if recursive else '僅限當前資料夾'}")
    print(f"========================================")

    # 依據參數決定是遍歷整個子目錄（rglob）還是只看當前目錄（glob）
    file_iterator = target_dir.rglob("*") if recursive else target_dir.glob("*")
    
    total_scanned = 0
    dirty_files_count = 0

    for file_path in file_iterator:
        # 只處理符合影像副檔名的檔案
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        
        total_scanned += 1
        has_metadata = False
        exif_to_print = {}
        png_info_to_print = {}

        try:
            with Image.open(file_path) as img:
                # 1. 檢查 JPEG 的 EXIF
                exif_data = img.getexif()
                if exif_data:
                    # 轉換 tag 名稱，並排除空的字典
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        exif_to_print[tag_name] = value
                    if exif_to_print:
                        has_metadata = True

                # 2. 檢查 PNG 的文字區塊 (Text Chunks)
                if img.format == "PNG" and img.info:
                    # 排除常見的軟體自動加上的無害 icc_profile 等幾何描述（可視需求調整）
                    # 如果你連 icc_profile 都要抓，可以直接：png_info_to_print = img.info
                    filtered_info = {k: v for k, v in img.info.items() if k not in ['icc_profile', 'interlace']}
                    if filtered_info:
                        png_info_to_print = filtered_info
                        has_metadata = True

                # 💡 核心邏輯：只有在偵測到有內藏資訊時，才輸出內容
                if has_metadata:
                    dirty_files_count += 1
                    print(f"\n🚨 [偵測到中繼資料] 檔案: {file_path.name}")
                    print(f"   相對路徑: {file_path}")
                    print(f"   格式: {img.format} | 尺寸: {img.size}")
                    
                    if exif_to_print:
                        print(f"   --- EXIF 詳情 ---")
                        for k, v in exif_to_print.items():
                            print(f"     📌 {k}: {v}")
                            
                    if png_info_to_print:
                        print(f"   --- PNG 內部文字 詳情 ---")
                        for k, v in png_info_to_print.items():
                            print(f"     📝 {k}: {v}")
                            
                    print("-" * 50)

        except Exception as e:
            # 略過損毀無法讀取的影像檔案
            pass

    print(f"\n========================================")
    print(f"📊 掃描報告完成")
    print(f"========================================")
    print(f"總計掃描影像檔案數: {total_scanned}")
    print(f"內藏中繼資料的檔案數: {dirty_files_count}")
    if dirty_files_count == 0:
        print("🎉 恭喜！所掃描的影像全部都很乾淨，無任何殘留中繼資料。")
    print(f"========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="大批次掃描指定資料夾，只有當影像內藏 EXIF 或中繼資料時才輸出統計資訊。"
    )

    # 目錄路徑參數
    parser.add_argument(
        "folder_path",
        type=str,
        nargs="?",
        default=".",  # 若沒輸入，預設掃描當前指令所在的資料夾
        help="輸入要掃描的資料夾路徑 (預設為當前目錄 '.')",
    )

    # 是否啟用遞迴掃描子目錄的旗標
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="啟用此旗標以深入掃描所有子資料夾下的影像",
    )

    args = parser.parse_args()
    scan_folder_metadata(args.folder_path, args.recursive)
    

import os
import shutil
import glob

def fix_structure():
    # 1. Định nghĩa đường dẫn chuẩn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, 'data', 'train')
    
    # Đây là 2 thư mục đích bắt buộc phải có
    target_real = os.path.join(data_root, 'real')
    target_fake = os.path.join(data_root, 'fake')

    print("="*60)
    print("ĐANG TỰ ĐỘNG SỬA CẤU TRÚC DỮ LIỆU...")
    print(f"Đích Real: {target_real}")
    print(f"Đích Fake: {target_fake}")
    print("="*60)

    # Luôn tạo lại thư mục đích để đảm bảo tồn tại
    os.makedirs(target_real, exist_ok=True)
    os.makedirs(target_fake, exist_ok=True)

    # 2. Quét toàn bộ thư mục 'data' để tìm ảnh bị lạc
    scan_dir = os.path.join(base_dir, 'data')
    
    moved_count = 0
    
    for root, dirs, files in os.walk(scan_dir):
        # Bỏ qua chính 2 folder đích để tránh copy lòng vòng
        if os.path.abspath(root) == os.path.abspath(target_real) or \
           os.path.abspath(root) == os.path.abspath(target_fake):
            continue

        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                src_path = os.path.join(root, file)
                path_lower = src_path.lower()
                
                #phân loại ảnh
                dest_folder = None
                
                # 1. Nhận diện ảnh REAL (người thật)
                if any(x in path_lower for x in ['real', 'live', 'selfie', 'nguoi_']): 
                    # Tránh nhầm lẫn nếu folder cha là fake
                    if not any(x in path_lower for x in ['fake', 'spoof', 'print', 'attack']):
                         dest_folder = target_real
                    elif 'real' in path_lower: # Ưu tiên từ khóa real nếu có cả 2
                         dest_folder = target_real

                # 2. Nhận diện ảnh FAKE (giả mạo)
                if any(x in path_lower for x in ['fake', 'spoof', 'attack', 'print', 'replay', 'mask', 'cut']):
                    dest_folder = target_fake

                if dest_folder:
                    try:
                        # Đổi tên file để tránh trùng: TênFolderCha_TênFile
                        parent = os.path.basename(root)
                        new_name = f"{parent}_{file}"
                        dst_path = os.path.join(dest_folder, new_name)
                        
                        # Chỉ di chuyển nếu chưa có ở đích
                        if not os.path.exists(dst_path):
                            shutil.move(src_path, dst_path)
                            moved_count += 1
                            if moved_count % 50 == 0: print(f"Đã chuyển {moved_count} ảnh về đúng chỗ...", end='\r')
                    except Exception as e:
                        pass

    print(f"\nĐÃ XONG! Tổng cộng sắp xếp lại: {moved_count} ảnh.")
    
    # Kiểm tra số lượng cuối cùng
    n_real = len(os.listdir(target_real))
    n_fake = len(os.listdir(target_fake))
    print(f"KẾT QUẢ HIỆN TẠI:")
    print(f"   - Real: {n_real} ảnh")
    print(f"   - Fake: {n_fake} ảnh")

    if n_real > 0 and n_fake > 0:
        print("\nCấu trúc đã chuẩn! Bạn hãy chạy lại file 'train_texture_cnn.py' ngay.")
    else:
        print("\nVẫn chưa tìm thấy ảnh. Hãy kiểm tra xem bạn đã tải dữ liệu về chưa?")

if __name__ == "__main__":
    fix_structure()

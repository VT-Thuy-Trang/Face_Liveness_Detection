import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.texture_cnn import LivenessNet
import os
import sys

def train():
    # 1. CẤU HÌNH ĐƯỜNG DẪN
    # Lấy đường dẫn của chính file code này đang chạy
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)

    # Tạo đường dẫn tuyệt đối tới folder data và models
    DATA_DIR = os.path.join(project_root, 'data', 'train')
    MODEL_DIR = os.path.join(project_root, 'models')
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'trained_model.pth')

    print("="*50)
    print(f"[INFO] Thư mục dự án: {project_root}")
    print(f"[INFO] Đang tìm dữ liệu tại: {DATA_DIR}")
    print(f"[INFO] Model sẽ được lưu tại: {MODEL_SAVE_PATH}")
    print("="*50)

    #2. KIỂM TRA DỮ LIỆU
    if not os.path.exists(DATA_DIR):
        print(f"\n[LỖI NGHIÊM TRỌNG] Không tìm thấy thư mục: {DATA_DIR}")
        print("Vui lòng kiểm tra lại xem bạn đã tạo folder 'data' và 'train' chưa?")
        print(f"Đường dẫn máy tính đang tìm kiếm là: {DATA_DIR}")
        return

    # Kiểm tra xem có folder con real/fake chưa
    if not os.path.exists(os.path.join(DATA_DIR, 'real')) or \
       not os.path.exists(os.path.join(DATA_DIR, 'fake')):
        print("\n[LỖI] Trong 'data/train' thiếu thư mục 'real' hoặc 'fake'.")
        return

    #3. CHUẨN BỊ DỮ LIỆU 
    # Resize về 32x32 để khớp với kiến trúc MiniVGG (LivenessNet)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except Exception as e:
        print(f"[LỖI] Không đọc được ảnh: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"[INFO] Tìm thấy {len(dataset)} ảnh.")
    print(f"[INFO] Classes: {dataset.class_to_idx}") # 0: fake, 1: real (hoặc ngược lại)

    #4. KHỞI TẠO MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Đang train trên thiết bị: {device}")
    
    model = LivenessNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #5. BẮT ĐẦU TRAINING
    EPOCHS = 20
    print("\n[START] Bắt đầu huấn luyện...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Tính độ chính xác
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(dataloader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    #6. LƯU KẾT QUẢ 
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("\n" + "="*50)
    print(f"[SUCCESS] Đã lưu model thành công tại: {MODEL_SAVE_PATH}")
    print("Bây giờ bạn có thể chạy file realtime_app.py!")
    print("="*50)

if __name__ == "__main__":
    train()
import torch
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def inspect_mambaout():
    print("Đang khởi tạo MambaOut-Femto từ timm")
    # Gọi mô hình và đổi lớp phân loại cuối thành 8 lớp (cho FERPlus)
    try:
        model = timm.create_model('mambaout_femto', pretrained=False, num_classes=8)
    except Exception as e:
        print("Không tìm thấy model")
        return

    model.eval()
    # Tạo một tensor ảnh giả lập độ phân giải 128x128
    dummy_input = torch.randn(1, 3, 128, 128)
    
    # In cấu trúc block/stage
    print("=== CẤU TRÚC MÔ HÌNH ===")
    # in toàn bộ mạng:
    # print(model) 
    
    # Đo Params và FLOPs bằng fvcore
    flops = FlopCountAnalysis(model, dummy_input)
    print("\n=== THỐNG KÊ CHI PHÍ ===")
    print(f"Tổng số tham số (Params): {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"Tổng số FLOPs: {flops.total() / 1e9:.2f} G (GigaFLOPs)")
    
if __name__ == '__main__':
    inspect_mambaout()
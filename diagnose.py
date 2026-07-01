"""DICOM 影像诊断脚本"""
import sys
import SimpleITK as sitk
import pydicom
import numpy as np

def diagnose(folder_path):
    print(f"诊断文件夹: {folder_path}")
    print("=" * 50)

    # 用 SimpleITK 加载
    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
    print(f"找到 {len(series_ids)} 个序列")

    if not series_ids:
        print("❌ 未找到 DICOM 序列")
        return

    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    print(f"序列切片数: {len(dicom_names)}")

    reader.SetFileNames(dicom_names)
    img = reader.Execute()

    print(f"图像尺寸: {img.GetSize()}")
    print(f"像素类型: {img.GetPixelIDTypeAsString()}")
    print(f"体素间距: {img.GetSpacing()}")

    arr = sitk.GetArrayFromImage(img)
    print(f"\n--- 原始数据 ---")
    print(f"数组形状: {arr.shape}")
    print(f"数据范围: {arr.min()} ~ {arr.max()}")
    print(f"数据类型: {arr.dtype}")
    print(f"中心切片均值: {arr[arr.shape[0]//2].mean():.2f}")

    # 检查 RescaleSlope/Intercept
    ds = pydicom.dcmread(str(dicom_names[0]), stop_before_pixels=True)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    print(f"\n--- DICOM 元数据 ---")
    print(f"RescaleSlope: {slope}")
    print(f"RescaleIntercept: {intercept}")
    print(f"Modality: {getattr(ds, 'Modality', 'unknown')}")

    if slope != 1.0 or intercept != 0.0:
        print(f"\n--- 应用 Rescale 后 ---")
        hu_array = arr.astype(np.float64) * slope + intercept
        print(f"HU 范围: {hu_array.min():.0f} ~ {hu_array.max():.0f}")
        print(f"中心切片 HU 均值: {hu_array[hu_array.shape[0]//2].mean():.2f}")
        print(f"空气区域 (< -900 HU): {np.sum(hu_array < -900)} 体素")
        print(f"软组织区域 (-200~200 HU): {np.sum((hu_array >= -200) & (hu_array <= 200))} 体素")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose(sys.argv[1])
    else:
        folder = input("请输入 DICOM 文件夹路径: ").strip()
        if folder:
            diagnose(folder)

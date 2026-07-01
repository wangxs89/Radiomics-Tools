"""ROI 处理模块"""
from typing import List, Optional
from dataclasses import dataclass
import pydicom


@dataclass
class ROI:
    """感兴趣区域"""
    number: int
    name: str
    contour_data: List[List[float]]


class ROIHandler:
    """ROI 处理器"""

    def load_rtstruct(self, file_path: str) -> Optional[List[ROI]]:
        """加载 RTSTRUCT 文件"""
        try:
            dataset = pydicom.dcmread(file_path)

            if dataset.Modality != "RTSTRUCT":
                print("不是 RTSTRUCT 文件")
                return None

            rois = []

            if hasattr(dataset, 'StructureSetROISequence'):
                for roi_seq in dataset.StructureSetROISequence:
                    roi_number = roi_seq.ROINumber
                    roi_name = roi_seq.ROIName

                    contour_data = []

                    if hasattr(dataset, 'ROIContourSequence'):
                        for contour_seq in dataset.ROIContourSequence:
                            if hasattr(contour_seq, 'ReferencedROINumber'):
                                if contour_seq.ReferencedROINumber == roi_number:
                                    if hasattr(contour_seq, 'ContourSequence'):
                                        for contour in contour_seq.ContourSequence:
                                            if hasattr(contour, 'ContourData'):
                                                points = contour.ContourData
                                                for i in range(0, len(points), 3):
                                                    if i + 2 < len(points):
                                                        contour_data.append([
                                                            float(points[i]),
                                                            float(points[i+1]),
                                                            float(points[i+2])
                                                        ])

                    roi = ROI(
                        number=roi_number,
                        name=roi_name,
                        contour_data=contour_data
                    )
                    rois.append(roi)

            return rois
        except Exception as e:
            print(f"加载 RTSTRUCT 失败: {e}")
            return None

    def get_roi_by_name(self, rois: List[ROI], name: str) -> Optional[ROI]:
        """按名称获取 ROI"""
        for roi in rois:
            if roi.name == name:
                return roi
        return None

    def get_roi_names(self, rois: List[ROI]) -> List[str]:
        """获取所有 ROI 名称"""
        return [roi.name for roi in rois]

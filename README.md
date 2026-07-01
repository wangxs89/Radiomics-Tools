# 影像组学 Web App

基于 Streamlit 的影像组学特征提取 Web 应用

## 功能特点

- 🟢 **初级模式**：一键上传，自动提取
- 🟡 **高级模式**：精细控制参数
- 📊 **可视化验证**：检查 ROI 位置
- 📈 **统计分析**：ICC、LASSO、相关性分析

## 安装

```bash
pip install -r requirements.txt
```

> **macOS ARM64 用户注意：** pyradiomics 需要额外安装 versioneer：
> ```bash
> pip install versioneer
> pip install pyradiomics --no-build-isolation
> ```

## 运行

```bash
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 推送代码到 GitHub
2. 访问 https://share.streamlit.io
3. 选择仓库和主文件（app.py）
4. 点击 Deploy

## 技术栈

- Streamlit
- PyRadiomics
- SimpleITK
- pydicom
- Plotly

## 许可证

MIT License

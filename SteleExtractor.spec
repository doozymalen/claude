# PyInstaller 설정: 더블클릭으로 실행하는 Windows 프로그램 만들기
#   pyinstaller --noconfirm SteleExtractor.spec
# 결과: dist/SteleExtractor/SteleExtractor.exe (폴더째로 배포)

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas, binaries, hiddenimports = [], [], []

for pkg in ("rapidocr_onnxruntime", "pyvista", "pyvistaqt", "trimesh", "ezdxf", "shapely"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("anthropic")
# VTK는 pyvista가 실제로 쓰는 모듈만 들어간다 (전부 넣으면 500MB 이상 커짐)
hiddenimports += ["vtkmodules.util.numpy_support", "vtkmodules.qt.QVTKRenderWindowInteractor"]
hiddenimports += ["scipy.optimize", "scipy.spatial", "scipy.ndimage", "mapbox_earcut", "onnxruntime"]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "jupyter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SteleExtractor",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SteleExtractor")

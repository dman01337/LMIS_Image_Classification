from PyInstaller.utils.hooks import collect_dynamic_libs

# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Classification_Script.py'],
    pathex=['C:\\Users\\daled\\Documents\\Flatiron\\Course_material\\Phase_5\\LMIS_Image_Classification'],
    binaries=collect_dynamic_libs('scipy'),
    datas=[
        ('C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/Models/Model_Multi_28.h5', '.'),
        ('C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/eye.ico', '.')
    ],
    hiddenimports=['tensorflow', 'pandas', 'scipy', 'scipy.libs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Classification_Script',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['eye.ico'],
)

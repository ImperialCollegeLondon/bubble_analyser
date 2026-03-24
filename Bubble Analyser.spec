# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\new_sizer\\bubble_analyser\\bubble_analyser\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('C:\\new_sizer\\bubble_analyser\\bubble_analyser\\config.toml', 'bubble_analyser'),
        ('C:\\new_sizer\\bubble_analyser\\bubble_analyser\\weights\\mask_rcnn_bubble.h5', 'bubble_analyser/weights'),
        ('C:\\new_sizer\\bubble_analyser\\bubble_analyser\\mrcnn', 'bubble_analyser/mrcnn'),
        ('C:\\new_sizer\\bubble_analyser\\bubble_analyser\\bubble', 'bubble_analyser/bubble'),
        ('C:\\new_sizer\\bubble_analyser\\.venv\\Lib\\site-packages\\tensorflow-plugins', 'tensorflow-plugins')
    ],
    hiddenimports=['scipy.special.cython_special', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils'],
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
    name='Bubble Analyser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

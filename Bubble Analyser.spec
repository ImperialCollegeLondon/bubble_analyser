# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/eeeyoung/Bubbles/bubble_analyser/bubble_analyser/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/eeeyoung/Bubbles/bubble_analyser/bubble_analyser/config.toml', 'bubble_analyser'), ('/Users/eeeyoung/Bubbles/bubble_analyser/bubble_analyser/weights/mask_rcnn_bubble.h5', 'bubble_analyser/weights'), ('/Users/eeeyoung/Bubbles/bubble_analyser/bubble_analyser/mrcnn', 'bubble_analyser/mrcnn'), ('/Users/eeeyoung/Bubbles/bubble_analyser/bubble_analyser/bubble', 'bubble_analyser/bubble')],
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
    [],
    exclude_binaries=True,
    name='Bubble Analyser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Bubble Analyser',
)
app = BUNDLE(
    coll,
    name='Bubble Analyser.app',
    icon=None,
    bundle_identifier=None,
)

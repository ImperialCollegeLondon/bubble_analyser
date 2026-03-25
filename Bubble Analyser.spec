# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# copy_metadata bundles the .dist-info folder for packages that call
# importlib.metadata.version() at import time (imgaug, imageio, etc.)
metadata_datas = []
for pkg in [
    'imageio',
    'imgaug',
    'numpy',
    'scipy',
    'matplotlib',
    'tqdm',
    'scikit-image',
    'Pillow',
    'h5py',
    'tensorflow-cpu',
    'tensorflow-directml-plugin',
    'tensorflow-io-gcs-filesystem',
]:
    try:
        metadata_datas += copy_metadata(pkg)
    except Exception:
        pass  # Skip if not installed (e.g. on a machine without that package)

a = Analysis(
    ['C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\__main__.py'],
    pathex=[],
    binaries=[
        # DirectML DLL must be in binaries (not datas) so it lands on the DLL search path
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\.venv\\Lib\\site-packages\\tensorflow-plugins\\directml\\*.dll', 'tensorflow-plugins/directml'),
    ],
    datas=[
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\config.toml', 'bubble_analyser'),
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\weights\\mask_rcnn_bubble.h5', 'bubble_analyser/weights'),
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\mrcnn', 'bubble_analyser/mrcnn'),
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\bubble', 'bubble_analyser/bubble'),
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\bubble_analyser\\methods', 'bubble_analyser/methods'),
        # tensorflow-plugins folder so TF can find the DirectML plugin at runtime
        ('C:\\Users\\23239560\\bubble\\bubble_analyser\\.venv\\Lib\\site-packages\\tensorflow-plugins', 'tensorflow-plugins'),
    ] + metadata_datas,
    hiddenimports=[
        'scipy.special.cython_special',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        # tensorflow-cpu on Windows ships the actual module via tensorflow-intel
        'tensorflow',
        'tensorflow_core',
        'tensorflow_directml_plugin',
    ],
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

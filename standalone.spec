# -*- mode: python ; coding: utf-8 -*-
block_cipher = None
data_files = [
    ('bubble_analyser/config.toml', 'bubble_analyser/'),
    ('bubble_analyser/methods/information.txt', 'bubble_analyser/methods/'),
    ('bubble_analyser/methods/watershed_methods.py', 'bubble_analyser/methods/')
]

a = Analysis(
    ['bubble_analyser/__main__.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cypher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bubble_analyser',
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

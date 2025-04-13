# -*- mode: python ; coding: utf-8 -*-
import platform
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

# For Windows build
if platform.system() == 'Windows':
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

# For macOS build
if platform.system() == 'Darwin':
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
        argv_emulation=True,  # Enable argv emulation for macOS
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    app = BUNDLE(
        exe,
        name='Bubble Analyser.app',
        icon=None,  # Add your .icns file path here if available
        bundle_identifier='com.imperial.bubble-analyser',
        version='0.2.0',  # Match version from pyproject.toml
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSPrincipalClass': 'NSApplication',
            'CFBundleName': 'Bubble Analyser',
            'CFBundleDisplayName': 'Bubble Analyser',
            'CFBundleGetInfoString': 'Bubble image analysis tool',
            'CFBundleShortVersionString': '0.2.0',
        },
        debug=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        argv_emulation=True,  # Enable argv emulation for macOS
        codesign_identity=None,  # Set to your Developer ID for distribution
        entitlements_file=None,
    )

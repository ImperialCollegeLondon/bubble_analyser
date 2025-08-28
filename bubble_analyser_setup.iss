[Setup]
AppName=Bubble Analyser
AppVersion=1.0
AppPublisher=Your Name
AppPublisherURL=https://github.com/yourusername/bubble_analyser
DefaultDirName={autopf}\BubbleAnalyser
DefaultGroupName=Bubble Analyser
OutputDir=dist
OutputBaseFilename=BubbleAnalyser_Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
; SetupIconFile=icon.ico
UninstallDisplayIcon={app}\bubble_analyser.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Main executable (assuming you've built it with PyInstaller)
Source: "dist\bubble_analyser.exe"; DestDir: "{app}"; Flags: ignoreversion
; Include all dependencies from dist folder
Source: "dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Configuration files
Source: "bubble_analyser\config.toml"; DestDir: "{app}\config"; Flags: ignoreversion
; Sample data (optional)
Source: "tests\sample_images\*"; DestDir: "{app}\samples"; Flags: ignoreversion recursesubdirs
Source: "tests\calibration_files\*"; DestDir: "{app}\calibration"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Bubble Analyser"; Filename: "{app}\bubble_analyser.exe"
Name: "{group}\{cm:UninstallProgram,Bubble Analyser}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Bubble Analyser"; Filename: "{app}\bubble_analyser.exe"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Bubble Analyser"; Filename: "{app}\bubble_analyser.exe"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\bubble_analyser.exe"; Description: "{cm:LaunchProgram,Bubble Analyser}"; Flags: nowait postinstall skipifsilent

[Registry]
; File associations (optional)
Root: HKCR; Subkey: ".bubble"; ValueType: string; ValueName: ""; ValueData: "BubbleAnalyserFile"
Root: HKCR; Subkey: "BubbleAnalyserFile"; ValueType: string; ValueName: ""; ValueData: "Bubble Analyser File"
Root: HKCR; Subkey: "BubbleAnalyserFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\bubble_analyser.exe,0"
Root: HKCR; Subkey: "BubbleAnalyserFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\bubble_analyser.exe"" ""%1"""
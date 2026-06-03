[Setup]
AppName=Bubble Analyser
AppVersion=2.0.0
AppPublisher=Dr Diego Mesa
AppPublisherURL=https://github.com/diegomesa/bubble_analyser
DefaultDirName={autopf}\Bubble Analyser 2.0
DefaultGroupName=Bubble Analyser 2.0
OutputDir=dist
OutputBaseFilename=BubbleAnalyser_Setup_v2
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
; SetupIconFile=icon.ico
UninstallDisplayIcon={app}\bubble_analyser.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "downloadweights"; Description: "Download Deep Learning weights (~250MB). Required for CNN methods."; GroupDescription: "Additional Components:"; Flags: checkedonce

[Files]
Source: "dist\Bubble Analyser\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Configuration files (already bundled in _internal but can be external for easy editing)
Source: "bubble_analyser\config.toml"; DestDir: "{app}\config"; Flags: ignoreversion
; Sample data
Source: "tests\sample_images\*"; DestDir: "{app}\samples"; Flags: ignoreversion recursesubdirs
Source: "tests\calibration_files\*"; DestDir: "{app}\calibration"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Bubble Analyser"; Filename: "{app}\bubble_analyser.exe"
Name: "{group}\{cm:UninstallProgram,Bubble Analyser}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Bubble Analyser"; Filename: "{app}\bubble_analyser.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\bubble_analyser.exe"; Description: "{cm:LaunchProgram,Bubble Analyser}"; Flags: nowait postinstall skipifsilent

[Registry]
; File associations (optional)
Root: HKCR; Subkey: ".bubble"; ValueType: string; ValueName: ""; ValueData: "BubbleAnalyserFile"
Root: HKCR; Subkey: "BubbleAnalyserFile"; ValueType: string; ValueName: ""; ValueData: "Bubble Analyser File"
Root: HKCR; Subkey: "BubbleAnalyserFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\bubble_analyser.exe,0"
Root: HKCR; Subkey: "BubbleAnalyserFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\bubble_analyser.exe"" ""%1"""

[Code]
var
  DownloadPage: TDownloadWizardPage;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax <> 0 then
    Log(Format('  Downloading... %d%%', [Progress * 100 / ProgressMax]));
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then begin
    if WizardIsTaskSelected('downloadweights') then begin
      DownloadPage.Clear;
      // Using the correct URL from the v0.3.0 release
      DownloadPage.Add('https://github.com/ImperialCollegeLondon/bubble_analyser/releases/download/v0.3.0/mask_rcnn_bubble.h5', 'mask_rcnn_bubble.h5', '');
      DownloadPage.Show;
      try
        try
          DownloadPage.Download;
          Result := True;
        except
          if GetExceptionMessage <> '' then
            MsgBox('Download failed: ' + GetExceptionMessage, mbError, MB_OK);
          Result := True; // Let installation continue even if download fails
        end;
      finally
        DownloadPage.Hide;
      end;
    end else
      Result := True;
  end else
    Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  DestDir: String;
begin
  if CurStep = ssPostInstall then begin
    if WizardIsTaskSelected('downloadweights') then begin
      DestDir := ExpandConstant('{app}\_internal\bubble_analyser\weights');
      if not DirExists(DestDir) then
        ForceDirectories(DestDir);

      if FileExists(ExpandConstant('{tmp}\mask_rcnn_bubble.h5')) then begin
        if not FileCopy(ExpandConstant('{tmp}\mask_rcnn_bubble.h5'), DestDir + '\mask_rcnn_bubble.h5', False) then
          MsgBox('Failed to copy weights to application directory.', mbError, MB_OK);
      end;
    end;
  end;
end;

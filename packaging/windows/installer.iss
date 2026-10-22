; Inno Setup script: per-user install (no admin rights), Start menu + optional desktop shortcut.
; Built by CI: iscc /DAppVersion=0.1.0 packaging\windows\installer.iss
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6F1B7C2A-5E2D-4C55-9C39-2B6E9F4F2A11}
AppName=Second Brain
AppVersion={#AppVersion}
AppPublisher=Second Brain
DefaultDirName={localappdata}\Programs\Second Brain
DefaultGroupName=Second Brain
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=Second-Brain-Setup
SetupIconFile=..\icons\icon.ico
UninstallDisplayIcon={app}\SecondBrain.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\SecondBrain\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Second Brain"; Filename: "{app}\SecondBrain.exe"
Name: "{autodesktop}\Second Brain"; Filename: "{app}\SecondBrain.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SecondBrain.exe"; Description: "Open Second Brain"; Flags: nowait postinstall skipifsilent

; Your library in %USERPROFILE%\.secondbrain is kept on uninstall.
